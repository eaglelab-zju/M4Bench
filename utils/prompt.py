import copy
import re

from .grounding import grounding_template_mapping


def prepare_object_states_prompt(sample, remove_image_tag, grounding_template):
    if remove_image_tag == True:
        question = sample['question']
    else:
        question = sample['question_add_image_token']
    options = sample['options']
    prompt = (
        f"{question} \n"
        f"A. {options[0]}\n"
        f"B. {options[1]}"
    )
    options_dict = {
        'A': options[0],
        'B': options[1]
    }
    return prompt, question, options_dict

def prepare_state_invariance_prompt(sample, remove_image_tag, grounding_template):
    if remove_image_tag == True:
        question = sample['question']
    else:
        question = sample['question_add_image_token']
    
    options = sample['options']
    prompt = (
        f"{question} \n"
        f"A. {options[0]}\n"
        f"B. {options[1]}"
    )
    options_dict = {
        'A': options[0],
        'B': options[1]
    }
    return prompt, question, options_dict

def prepare_detailed_difference_prompt(sample, remove_image_tag, grounding_template):
    if remove_image_tag == True:
        question = sample['question']
    else:
        question = sample['question_add_image_token']
    
    option_labels = ['A', 'B', 'C', 'D']
    options = sample['options']
    options_dict = {}
    filled_options = []
    
    if grounding_template == 'closed_source':
        question = '[x1, y1, x2, y2] represents a bounding box, where (x1, y1) denotes the top-left corner and (x2, y2) denotes the bottom-right corner. ' + question
    grounding_func = grounding_template_mapping.get(grounding_template)
    if grounding_func is None:
        raise ValueError(f"Unsupported grounding template: {grounding_template}")
    
    for idx, option in enumerate(options):
        label = option_labels[idx]
        bbox_matches = re.findall(r'<([A-Z])_bbox>', option)
        
        if bbox_matches:
            bbox_label = bbox_matches[0]
            bbox = sample.get(f"{bbox_label}_bbox")
            
            left_size = sample.get('left_image_size')
            right_size = sample.get('right_image_size')
            
            left_box_str = grounding_func(
                bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'],
                left_size['width'], left_size['height']
            )
            right_box_str = grounding_func(
                bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'],
                right_size['width'], right_size['height']
            )
            
            option_replaced = option.replace(f"<{bbox_label}_bbox>", left_box_str, 1)
            option_replaced = option_replaced.replace(f"<{bbox_label}_bbox>", right_box_str, 1)
            filled_options.append(f"{label}. {option_replaced}")
            options_dict[label] = option_replaced

        else:
            # No difference
            filled_options.append(f"{label}. {option}")
            options_dict[label] = option
        
    prompt = f"{question}\n" + "\n".join(filled_options)
    return prompt, question, options_dict

def prepare_spatial_perception_with_visual_prompt_prompt(sample, remove_image_tag, grounding_template):
    if remove_image_tag:
        question = sample.get('question')
    else:
        raise NotImplementedError("remove_image_tag must be True for Spatial_Perception task")
    
    if grounding_template == 'closed_source':
        question = '[x1, y1, x2, y2] represents a bounding box, where (x1, y1) denotes the top-left corner and (x2, y2) denotes the bottom-right corner. ' + question

    grounding_func = grounding_template_mapping.get(grounding_template)
    if grounding_func is None:
        raise ValueError(f"Unsupported grounding template: {grounding_template}")
    
    left_size = sample.get('left_image_size', {})
    right_size = sample.get('right_image_size', {})

    question_bbox = sample.get('question_bbox', {})
    question_bbox_matches = re.findall(r'<bbox:(\d+)>', question)
    for bbox_id in question_bbox_matches:
        bbox_key = f"<bbox:{bbox_id}>"
        bbox_data = question_bbox.get(bbox_key)
        if not bbox_data:
            raise ValueError(f"Missing bounding box data for {bbox_key} in question_bbox")
        
        if bbox_id == '1':
            image_size = left_size
        elif bbox_id == '2':
            image_size = right_size
        else:
            raise ValueError(f"Unsupported bbox id in question: {bbox_id}")
        
        grounded_bbox = grounding_func(
            bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2'],
            image_size.get('width', 0), image_size.get('height', 0),
        )
        question = question.replace(bbox_key, grounded_bbox)

    option_labels = ['A', 'B', 'C', 'D']
    options = sample.get('options', [])
    options_dict = {}
    filled_options = []
    
    for idx, option in enumerate(options):
        label = option_labels[idx]
        current_option = option
        
        bbox_matches = re.findall(r'<([A-Z])_bbox:(\d+)>', option)
        
        for match in bbox_matches:
            bbox_label, bbox_id = match
            bbox_key = f"{bbox_label}_bbox"
            bbox_data = sample.get(bbox_key, {}).get(f"<bbox:{bbox_id}>")
            if not bbox_data:
                raise ValueError(f"Missing bounding box data for {bbox_key} and <bbox:{bbox_id}>")
            
            if bbox_id == '1':
                image_size = left_size
            elif bbox_id == '2':
                image_size = right_size
            else:
                raise ValueError(f"Unsupported bbox id: {bbox_id}")
            
            grounded_bbox = grounding_func(
                bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2'],
                image_size.get('width', 0), image_size.get('height', 0)
            )
            
            placeholder = f"<{bbox_label}_bbox:{bbox_id}>"
            current_option = current_option.replace(placeholder, grounded_bbox)
        
        filled_option = f"{label}. {current_option}"
        filled_options.append(filled_option)
        options_dict[label] = current_option
    
    prompt = f"{question}\n" + "\n".join(filled_options)
    return prompt, question, options_dict
    
def prepare_spatial_perception_without_visual_prompt_prompt(sample, remove_image_tag, grounding_template):
    if remove_image_tag:
        question = sample.get('question_without_prompt')
    else:
        raise NotImplementedError("remove_image_tag must be True for Spatial_Perception task")
    
    if grounding_template == 'closed_source':
        question = '[x1, y1, x2, y2] represents a bounding box, where (x1, y1) denotes the top-left corner and (x2, y2) denotes the bottom-right corner. ' + question

    grounding_func = grounding_template_mapping.get(grounding_template)
    if grounding_func is None:
        raise ValueError(f"Unsupported grounding template: {grounding_template}")
    
    left_size = sample.get('left_image_size', {})
    right_size = sample.get('right_image_size', {})

    option_labels = ['A', 'B', 'C', 'D']
    options = sample.get('options', [])
    options_dict = {}
    filled_options = []
    
    for idx, option in enumerate(options):
        label = option_labels[idx]
        current_option = option
        
        bbox_matches = re.findall(r'<([A-Z])_bbox:(\d+)>', option)
        
        for match in bbox_matches:
            bbox_label, bbox_id = match
            bbox_key = f"{bbox_label}_bbox"
            bbox_data = sample.get(bbox_key, {}).get(f"<bbox:{bbox_id}>")
            if not bbox_data:
                raise ValueError(f"Missing bounding box data for {bbox_key} and <bbox:{bbox_id}>")
            
            if bbox_id == '1':
                image_size = left_size
            elif bbox_id == '2':
                image_size = right_size
            else:
                raise ValueError(f"Unsupported bbox id: {bbox_id}")
            
            grounded_bbox = grounding_func(
                bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2'],
                image_size.get('width', 0), image_size.get('height', 0)
            )
            
            placeholder = f"<{bbox_label}_bbox:{bbox_id}>"
            current_option = current_option.replace(placeholder, grounded_bbox)
        
        filled_option = f"{label}. {current_option}"
        filled_options.append(filled_option)
        options_dict[label] = current_option
    
    prompt = f"{question}\n" + "\n".join(filled_options)
    return prompt, question, options_dict

def prepare_instance_comparison_with_visual_prompt_prompt(sample, remove_image_tag, grounding_template):
    if remove_image_tag:
        question = sample.get('question')
    else:
        raise NotImplementedError("remove_image_tag must be True for Instance_Comparison task")
    
    if grounding_template == 'closed_source':
        question = '[x1, y1, x2, y2] represents a bounding box, where (x1, y1) denotes the top-left corner and (x2, y2) denotes the bottom-right corner. ' + question

    grounding_func = grounding_template_mapping.get(grounding_template)
    if grounding_func is None:
        raise ValueError(f"Unsupported grounding template: {grounding_template}")
    
    left_size = sample.get('left_image_size', {})
    right_size = sample.get('right_image_size', {})

    question_bbox = sample.get('question_bbox', {})
    question_bbox_matches = re.findall(r'<bbox:(\d+)>', question)
    for bbox_id in question_bbox_matches:
        bbox_key = f"<bbox:{bbox_id}>"
        bbox_data = question_bbox.get(bbox_key)
        if not bbox_data:
            raise ValueError(f"Missing bounding box data for {bbox_key} in question_bbox")
        
        if bbox_id == '1':
            image_size = left_size
        elif bbox_id == '2':
            image_size = right_size
        else:
            raise ValueError(f"Unsupported bbox id in question: {bbox_id}")
        
        grounded_bbox = grounding_func(
            bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2'],
            image_size.get('width', 0), image_size.get('height', 0),
        )
        question = question.replace(bbox_key, grounded_bbox)

    option_labels = ['A', 'B', 'C', 'D']
    options = sample.get('options', [])
    options_dict = {}
    filled_options = []
    
    for idx, option in enumerate(options):
        label = option_labels[idx]
        current_option = option
        
        bbox_matches = re.findall(r'<([A-Z])_bbox:(\d+)>', option)
        
        for match in bbox_matches:
            bbox_label, bbox_id = match
            bbox_key = f"{bbox_label}_bbox"
            bbox_data = sample.get(bbox_key, {}).get(f"<bbox:{bbox_id}>")
            if not bbox_data:
                raise ValueError(f"Missing bounding box data for {bbox_key} and <bbox:{bbox_id}>")
            
            if bbox_id == '1':
                image_size = left_size
            elif bbox_id == '2':
                image_size = right_size
            else:
                raise ValueError(f"Unsupported bbox id: {bbox_id}")
            
            grounded_bbox = grounding_func(
                bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2'],
                image_size.get('width', 0), image_size.get('height', 0)
            )
            
            placeholder = f"<{bbox_label}_bbox:{bbox_id}>"
            current_option = current_option.replace(placeholder, grounded_bbox)
        
        filled_option = f"{label}. {current_option}"
        filled_options.append(filled_option)
        options_dict[label] = current_option
    
    prompt = f"{question}\n" + "\n".join(filled_options)
    return prompt, question, options_dict

def prepare_instance_comparison_without_visual_prompt_prompt(sample, remove_image_tag, grounding_template):
    if remove_image_tag:
        question = sample.get('question_without_prompt')
    else:
        raise NotImplementedError("remove_image_tag must be True for Instance_Comparison task")
    
    if grounding_template == 'closed_source':
        question = '[x1, y1, x2, y2] represents a bounding box, where (x1, y1) denotes the top-left corner and (x2, y2) denotes the bottom-right corner. ' + question
    
    grounding_func = grounding_template_mapping.get(grounding_template)
    if grounding_func is None:
        raise ValueError(f"Unsupported grounding template: {grounding_template}")
    
    left_size = sample.get('left_image_size', {})
    right_size = sample.get('right_image_size', {})

    option_labels = ['A', 'B', 'C', 'D']
    options = sample.get('options', [])
    options_dict = {}
    filled_options = []
    
    for idx, option in enumerate(options):
        label = option_labels[idx]
        current_option = option
        
        bbox_matches = re.findall(r'<([A-Z])_bbox:(\d+)>', option)
        
        for match in bbox_matches:
            bbox_label, bbox_id = match
            bbox_key = f"{bbox_label}_bbox"
            bbox_data = sample.get(bbox_key, {}).get(f"<bbox:{bbox_id}>")
            if not bbox_data:
                raise ValueError(f"Missing bounding box data for {bbox_key} and <bbox:{bbox_id}>")
            
            if bbox_id == '1':
                image_size = left_size
            elif bbox_id == '2':
                image_size = right_size
            else:
                raise ValueError(f"Unsupported bbox id: {bbox_id}")
            
            grounded_bbox = grounding_func(
                bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2'],
                image_size.get('width', 0), image_size.get('height', 0)
            )
            
            placeholder = f"<{bbox_label}_bbox:{bbox_id}>"
            current_option = current_option.replace(placeholder, grounded_bbox)
        
        filled_option = f"{label}. {current_option}"
        filled_options.append(filled_option)
        options_dict[label] = current_option
    
    prompt = f"{question}\n" + "\n".join(filled_options)
    return prompt, question, options_dict

# task_name : function
prompt_process_mapping = {
    "Object_States": prepare_object_states_prompt,
    "Detailed_Difference_Generated_Images": prepare_detailed_difference_prompt,
    "Detailed_Difference_Natural_Images": prepare_detailed_difference_prompt,
    "State_Invariance": prepare_state_invariance_prompt,
    "Spatial_Perception_With_Visual_Prompt": prepare_spatial_perception_with_visual_prompt_prompt,
    "Spatial_Perception_WithOut_Visual_Prompt": prepare_spatial_perception_without_visual_prompt_prompt,
    "Instance_Comparison_With_Visual_Prompt": prepare_instance_comparison_with_visual_prompt_prompt,
    "Instance_Comparison_WithOut_Visual_Prompt": prepare_instance_comparison_without_visual_prompt_prompt,
}