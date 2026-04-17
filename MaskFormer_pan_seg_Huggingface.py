@torch.no_grad()
def visualize_maskformer_predictions_final(
    model, dataset, device="cuda",
    score_threshold=0.5, num_images=3
):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from PIL import Image
    import torch
    import os
    import random
    import torch.nn.functional as F

    # COLORS
    COLOR_CAP1   = np.array([0, 100, 0])
    COLOR_CAP2   = np.array([255, 200, 0])
    COLOR_MARKER = np.array([128, 0, 255])
    COLOR_FLOOR  = np.array([135, 206, 235])

    model.eval()
    model.to(device)

    plt.figure(figsize=(6 * num_images, 6))

    id2label = model.config.id2label

    for idx in range(num_images):
        random_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[random_idx]

        pixel_values = sample['pixel_values'].unsqueeze(0).to(device)

        # Load original image
        filename = dataset.files[random_idx]
        img_path = os.path.join(dataset.image_root, filename)
        orig = np.array(Image.open(img_path).convert("RGB"))
        H, W = orig.shape[:2]

        overlay = orig.copy()

        # -------------------------
        # MODEL
        # -------------------------
        outputs = model(pixel_values=pixel_values)

        logits = outputs.class_queries_logits[0]
        masks_logits = outputs.masks_queries_logits[0]

        probs = logits.softmax(-1)[:, :-1]
        scores, labels = probs.max(-1)

        keep = scores > score_threshold

        scores = scores[keep]
        labels = labels[keep]
        masks_logits = masks_logits[keep]

        # -------------------------
        # RESIZE MASKS
        # -------------------------
        masks_logits = F.interpolate(
            masks_logits.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )[0]

        masks = masks_logits.sigmoid().cpu().numpy()

        alpha = 0.7

        # -------------------------
        # BUILD PANOPTIC MAP
        # -------------------------
        panoptic_map = np.zeros((H, W), dtype=np.int32)

        score_order = np.argsort(-scores.cpu().numpy())

        instance_id = 1
        cap_instances = []

        for i in score_order:
            mask = masks[i] > 0.5
            cls = labels[i].item()

            if mask.sum() == 0:
                continue

            if cls == 0:  # CAP
                panoptic_map[mask] = 1000 + instance_id
                cap_instances.append((instance_id, mask, scores[i].item()))
                instance_id += 1

            elif cls == 1:  # Marker
                panoptic_map[mask] = 2000

        # Fill remaining as FLOOR
        panoptic_map[panoptic_map == 0] = 3000

        # -------------------------
        # COLOR OVERLAY (PANOPTIC)
        # -------------------------
        for val in np.unique(panoptic_map):
            mask = panoptic_map == val

            class_id = val // 1000
            inst_id = val % 1000

            if class_id == 1:  # CAP
                color = COLOR_CAP1 if inst_id % 2 == 0 else COLOR_CAP2

            elif class_id == 2:  # Marker
                color = COLOR_MARKER

            elif class_id == 3:  # Floor
                color = COLOR_FLOOR

            else:
                continue

            overlay[mask] = (
                alpha * color + (1 - alpha) * overlay[mask]
            ).astype(np.uint8)

        # -------------------------
        # DRAW BOX + LABEL (ONLY CAP)
        # -------------------------
        for inst_id, mask, score in cap_instances:
            ys, xs = np.where(mask)

            if len(xs) == 0:
                continue

            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()

            color = COLOR_CAP1 if inst_id % 2 == 0 else COLOR_CAP2

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 3)

            label_name = id2label[0]

            cv2.putText(
                overlay,
                f"{label_name} {score:.2f}",
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 0, 0),
                4,
                cv2.LINE_AA
            )

        # -------------------------
        # SHOW
        # -------------------------
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(overlay)
        plt.axis("off")

    plt.tight_layout()
    plt.show()