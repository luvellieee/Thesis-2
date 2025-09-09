

def proc_ex(ex, tag_field = "tags_knowledge"):
    label = tag_field.split("_")[1]
    spans = []
    start_pos = None
    entity_name = None
    sentence = ""
    results = []
    prev_tag = "O"

    for i, (token, tag) in enumerate(zip(ex['tokens'], ex[tag_field])):
        if sentence == "":
            sentence = token
        else:
            sentence += " " + token
        
        if tag == "O":  # 'O' tag
            if entity_name is not None:
                results.append({"label": entity_name, "start": start, "end": end, 'text': text})
                spans.append((start_pos, i - 1))
                entity_name = None
                start_pos = None
                
            prev_tag = tag
        else:
            tag_name = tag
            if tag_name.startswith('B'):
                if entity_name is not None:
                    spans.append((start_pos, i - 1))
                    results.append({"label": entity_name, "start": start, "end": end, 'text': text})
                    
                entity_name = label.capitalize()
                start_pos = i

                start = len(sentence) - len(token)
                end = len(sentence)
                text = token

                prev_tag = tag
                
            elif tag_name.startswith('I'):
                if prev_tag == "O":
                    # one error case 
                    prev_token = ex['tokens'][i-1]
                    start = len(sentence) - (1 +len(token)) - len(prev_token)
                    text = prev_token
                    ner_start = i - 1
                    
                end = len(sentence)
                text += " " + token
                
                prev_tag = tag
                continue

    # Handle the last entity if the sentence ends with an entity
    if entity_name is not None:
        spans.append((start_pos, len(ex["tokens"]) - 1))
        results.append({"label": entity_name, "start": start, "end": end, 'text': text})

    return {"tokenized_text": ex["tokens"], 'sentence': sentence, f"{label}_ner": spans, label: results}
    
def formatting_prompts_func(ex):
    d1 = proc_ex(ex, tag_field="tags_knowledge")
    d2 = proc_ex(ex, tag_field="tags_skill")
    ex.update(d1)
    ex.update(d2)
    return ex


def convert_to_gliner_dataset(hg_data):
    res = []
    for tks, skill_ner, knowledge_ner in zip(hg_data['tokenized_text'], hg_data['skill_ner'], hg_data['knowledge_ner']):
        d = {"tokenized_text": tks}
        if len(skill_ner) == 0 and len(knowledge_ner) == 0:
            # The information tag is added when no entities are present, this is done to improve training but this tag will not be used
            d['ner'] = [[0, len(tks), "Information"]]
            res.append(d)
            continue
        d['ner'] = [[i,j, "Skill"] for i, j in skill_ner]
        d['ner'] += [[i,j, "Knowledge"] for i, j in knowledge_ner]
        res.append(d)
    return res


def combine_entities(examples):
    return {'knowledge_and_skill': examples['knowledge'] + examples['skill']}
