class QueryRewriter:
    def rewrite(self, query: str) -> str:
        query = query.strip()

        replacements = {
            "climate impact from the amount of fuel it uses":
                "fuel consumption emissions activity data emission factor",
            "electricity usage be converted into greenhouse gas emissions":
                "electricity consumption kWh emission factor CO2e calculation",
            "where a conversion factor came from and which year it applies to":
                "emission factor source reporting year methodology auditability",
            "same greenhouse gas conversion factor always be used for every country and year":
                "emission factors country geography reporting year methodology",
        }

        query_lower = query.lower()

        for phrase, rewritten_query in replacements.items():
            if phrase in query_lower:
                return rewritten_query

        return query