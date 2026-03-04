import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path

BRUSH_DIR = Path(__file__).parent / "brush"


def load_meta_brushes(device="cuda"):
    v = _read_brush(BRUSH_DIR / "brush_large_vertical.png", device)
    h = _read_brush(BRUSH_DIR / "brush_large_horizontal.png", device)
    return torch.cat([v, h], dim=0)


def _read_brush(path, device):
    img = Image.open(path).convert("L")
    t = torch.from_numpy(np.array(img)).unsqueeze(0).unsqueeze(0).float() / 255.0
    return t.to(device)


def _erosion(x, m=1):
    b, c, h, w = x.shape
    x_pad = F.pad(x, [m, m, m, m], mode="constant", value=1e9)
    channel = F.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    return torch.min(channel, dim=2)[0]


def _dilation(x, m=1):
    b, c, h, w = x.shape
    x_pad = F.pad(x, [m, m, m, m], mode="constant", value=-1e9)
    channel = F.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    return torch.max(channel, dim=2)[0]


def param2stroke(param, H, W, meta_brushes):
    meta_brushes_resize = F.interpolate(meta_brushes, (H, W))
    b = param.shape[0]
    param_list = torch.split(param, 1, dim=1)
    x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
    R, G, B = param_list[5:]

    sin_theta = torch.sin(torch.acos(torch.tensor(-1.0, device=param.device)) * theta)
    cos_theta = torch.cos(torch.acos(torch.tensor(-1.0, device=param.device)) * theta)

    index = torch.full((b,), -1, device=param.device, dtype=torch.long)
    index[h > w] = 0
    index[h <= w] = 1
    brush = meta_brushes_resize[index.long()]

    warp_00 = cos_theta / w
    warp_01 = sin_theta * H / (W * w)
    warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
    warp_10 = -sin_theta * W / (H * h)
    warp_11 = cos_theta / h
    warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
    warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
    warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
    warp = torch.stack([warp_0, warp_1], dim=1)

    grid = F.affine_grid(warp, [b, 3, H, W], align_corners=False)
    brush = F.grid_sample(brush, grid, align_corners=False)
    alphas = (brush > 0).float()
    brush = brush.repeat(1, 3, 1, 1)
    alphas = alphas.repeat(1, 3, 1, 1)

    color_map = torch.cat([R, G, B], dim=1)
    color_map = color_map.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
    foreground = brush * color_map
    foreground = _dilation(foreground)
    alphas = _erosion(alphas)
    return foreground, alphas


def render_strokes(params, canvas_h=256, canvas_w=256, canvas=None, device="cuda", meta_brushes=None):
    if meta_brushes is None:
        meta_brushes = load_meta_brushes(device)
    if canvas is None:
        canvas = torch.ones(1, 3, canvas_h, canvas_w, device=device)
    if params.dim() == 2:
        params = params.unsqueeze(0)

    b = canvas.shape[0]
    for i in range(b):
        strokes = params[i]
        if strokes.shape[0] == 0:
            continue
        foregrounds, alphas = param2stroke(strokes, canvas_h, canvas_w, meta_brushes)
        for j in range(strokes.shape[0]):
            fg = foregrounds[j:j+1]
            a = alphas[j:j+1]
            canvas[i:i+1] = fg * a + canvas[i:i+1] * (1 - a)
    return canvas


def render_to_image(params, canvas_h=256, canvas_w=256, device="cuda"):
    canvas = render_strokes(params, canvas_h, canvas_w, device=device)
    img = canvas[0].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray((img * 255).astype(np.uint8))


def parse_stroke_string(text):
    strokes = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            vals = [float(v.strip()) for v in line.split(",")]
            if len(vals) == 8:
                strokes.append(vals)
        except (ValueError, IndexError):
            continue
    if not strokes:
        return None
    return torch.tensor(strokes, dtype=torch.float32)
