@app.tool(output="ans_ls->extract_query_list")
def search_o1_query_extract(ans_ls: List[str]) -> Dict[str, List[str]]:

    def get_query(text):
        import re
        pattern = (
            re.escape("<|begin_search_query|>")
            + r"(.*?)"
            + re.escape("<|end_search_query|>")
        )
        matches = re.findall(pattern, text, flags=re.DOTALL)

        if matches:
            query = matches[-1].strip()
            if not query.endswith("?"):
                query += "?"
            return query
        else:
            return "There is no query."

    query = [get_query(answer) for answer in ans_ls]

    return {"extract_query_list": query}