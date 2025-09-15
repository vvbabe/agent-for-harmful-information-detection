@app.tool(output="ans_ls->pred_ls")
def output_extract_from_boxed(ans_ls: List[str]) -> Dict[str, List[str]]:
    #大模型生成的文本往往包含「推理过程 + 最终答案」，而且格式不固定,如果我们要求模型用 \boxed{...} 包裹最终答案，那么无论推理过程多复杂，最终答案都有一个固定标记，方便后续提取。
    #在很多benchmark里也使用boxed{答案}
    def extract(ans: str) -> str:
        start = ans.rfind(r"\boxed{")
        if start == -1:
            #如果没找到boxed,说明大模型没找到答案
            content = ans.strip()
        else:
            i = start + len(r"\boxed{")
            brace_level = 1
            end = i

            #数嵌套的层数以免出错，取最外层嵌套
            while end < len(ans) and brace_level > 0:
                if ans[end] == "{":
                    brace_level += 1
                elif ans[end] == "}":
                    brace_level -= 1
                end += 1
            content = ans[i : end - 1].strip()
            content = re.sub(r"^\$+|\$+$", "", content).strip()
            content = re.sub(r"^\\\(|\\\)$", "", content).strip()
            if content.startswith(r"\text{") and content.endswith("}"):
                content = content[len(r"\text{") : -1].strip()
            content = content.strip("()").strip()
        # 还原 \\
        content = content.replace("\\", " ")
        content = content.replace("  ", " ")
        return content

    return {"pred_ls": [extract(ans) for ans in ans_ls]}