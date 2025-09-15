@app.prompt(output="q_ls,ret_psg,template->prompt_ls")
def qa_rag_boxed(
    #q_ls问题列表，ret_psg检索到的信息，template:模板
    
    q_ls:List[str],ret_psg:List[str | Any],template:str | Path
)->list[PromptMessage]:
    template:Template = load_prompt_template(template)
    ret=[]
    #把每个问题对应的检索结果整合成一段文本
    #用模板渲染成标准化prompt
    for q,psg in zip(q_ls,ret_psg):
        passage_text="\n".join(psg)
        p=template.render(question=q,documents=passage_text)
        ret.append(p)
    return ret#这返回的是渲染好的prompt列表


@app.prompt(output="q_ls,ret_psg,template2->prompt_ls")
def qa_rag_boxed_2(
    q_ls: List[str], ret_psg: List[str | Any], template: str | Path
) -> list[PromptMessage]: