def qwen2vl_grounding_template_func(x1, y1, x2, y2, image_width, image_height, max_range=1000):
    x_scale, y_scale = max_range / image_width, max_range / image_height
    x1, x2 = x1 * x_scale, x2 * x_scale
    y1, y2 = y1 * y_scale, y2 * y_scale
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    return f"<|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"

def internvl2_grounding_template_func(x1, y1, x2, y2, image_width, image_height, max_range=1000):
    x1, y1, x2, y2 = [
        round((x1 / image_width) * max_range),
        round((y1 / image_height) * max_range),
        round((x2 / image_width) * max_range),
        round((y2 / image_height) * max_range)
    ]
    return f"<box>[[{x1}, {y1}, {x2}, {y2}]]</box>"

def deepseek_vl2_grounding_template_func(x1, y1, x2, y2, image_width, image_height, max_range=999):
    x_scale, y_scale = max_range / image_width, max_range / image_height
    x1, x2 = x1 * x_scale, x2 * x_scale
    y1, y2 = y1 * y_scale, y2 * y_scale
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    return f"<|det|>[[{x1}, {y1}, {x2}, {y2}]]<|/det|>"

def closed_source_grounding_template_func(x1, y1, x2, y2, image_width, image_height, max_range=None):
    return f"[{x1}, {y1}, {x2}, {y2}]"

# grounding_template: function
grounding_template_mapping = {
    "qwen2vl": qwen2vl_grounding_template_func,
    'internvl2': internvl2_grounding_template_func,
    'deepseek-vl2': deepseek_vl2_grounding_template_func,
    'closed_source': closed_source_grounding_template_func,
}