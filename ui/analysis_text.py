"""Present known internal asset references as the same labels used in the sidebar."""
def display_analysis_text(text, metadata):
    if not isinstance(text,str):return text
    for index,info in enumerate(metadata.values(),1):
        text=text.replace(info.id,f'결과 {index}')
    return text
