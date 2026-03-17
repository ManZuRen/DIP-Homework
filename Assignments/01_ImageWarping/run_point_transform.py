import cv2
import numpy as np
import gradio as gr


# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None


# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img


# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image


# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """
    if image is None:
        return None

    warped_image = np.array(image)
    num_pairs = min(len(source_pts), len(target_pts))
    if num_pairs == 0:
        return warped_image
    if len(source_pts) != len(target_pts):
        raise ValueError("源点和目标点数量必须一致，请成对选择控制点。")
    if num_pairs < 3:
        raise ValueError("MLS 形变至少需要 3 对控制点。")

    source_pts = np.asarray(source_pts[:num_pairs], dtype=np.float32)
    target_pts = np.asarray(target_pts[:num_pairs], dtype=np.float32)

    # MLS affine deformation with backward mapping:
    # for each pixel on the output image, estimate a local affine transform
    # from target control points back to source control points.
    p = source_pts
    q = target_pts

    height, width = warped_image.shape[:2]
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    identity = np.eye(2, dtype=np.float32)

    for y in range(height):
        row_points = np.stack(
            [np.arange(width, dtype=np.float32), np.full(width, y, dtype=np.float32)],
            axis=1,
        )

        diff = row_points[:, None, :] - q[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        exact_mask = dist_sq < eps

        weights = 1.0 / np.maximum(dist_sq, eps) ** alpha
        weight_sum = np.sum(weights, axis=1, keepdims=True)

        q_star = (weights @ q) / weight_sum
        p_star = (weights @ p) / weight_sum

        q_hat = q[None, :, :] - q_star[:, None, :]
        p_hat = p[None, :, :] - p_star[:, None, :]

        m_matrix = np.einsum("wn,wni,wnj->wij", weights, q_hat, q_hat)
        m_matrix += eps * identity[None, :, :]
        b_matrix = np.einsum("wn,wni,wnj->wij", weights, p_hat, q_hat)

        try:
            inv_m = np.linalg.inv(m_matrix)
        except np.linalg.LinAlgError:
            inv_m = np.linalg.pinv(m_matrix)

        affine = np.matmul(b_matrix, inv_m)
        v_hat = row_points - q_star
        mapped = np.einsum("wji,wi->wj", affine, v_hat) + p_star

        if np.any(exact_mask):
            matched_rows = np.where(np.any(exact_mask, axis=1))[0]
            matched_ctrl = np.argmax(exact_mask[matched_rows], axis=1)
            mapped[matched_rows] = p[matched_ctrl]

        map_x[y] = mapped[:, 0]
        map_y[y] = mapped[:, 1]
    warped_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    return warped_image


def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image


# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image


# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
