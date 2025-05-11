from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, Schema
import json

class TradingRecommender:
    def __init__(self):
        with open('.api_key.json', 'r') as f:
            api_key = json.load(f)['api_key']
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.search_tool = Tool(google_search=GoogleSearch())

    def get_recommendations(self, sector, return_json=True):
        """
        Get stock recommendations for a specific sector using Gemini API with structured JSON output.

        Args:
            sector (str): The sector to get stock recommendations for
            return_json (bool): Whether to return the recommendations as JSON (default: True)

        Returns:
            List of stock recommendations as dictionaries or prints the text response
        """
        # Step 1: Use Google Search to get high-quality information
        search_query = f"Which stocks in the {sector} sector are performing well and are good long term investments? Research current market trends and analyst recommendations."

        try:
            # First call with search tools to get information
            search_response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=search_query,
                config=GenerateContentConfig(
                    tools=[self.search_tool],
                    response_modalities=['TEXT']
                )
            )

            # Extract the search results
            search_results = ""
            for part in search_response.candidates[0].content.parts:
                search_results += part.text

            # Step 2: Format the information into structured JSON
            if return_json:
                # Define the schema for structured output
                stock_schema = Schema(
                    type="ARRAY",
                    items=Schema(
                        type="OBJECT",
                        properties={
                            "ticker": Schema(type="STRING"),
                            "name": Schema(type="STRING"),
                            "rationale": Schema(type="STRING")
                        },
                        required=["ticker", "name", "rationale"]
                    )
                )

                # Format query for structured output
                json_query = f"""Based on this research about {sector} sector stocks:

                {search_results}

                Extract 3-5 of the most promising stocks and format them as a JSON array.
                Each stock should be an object with these exact fields:
                - ticker: The stock symbol
                - name: The company name
                - rationale: A concise but informative explanation of why it's a good investment

                Only return valid JSON with no additional text."""

                # Second call to get structured JSON (without search tools)
                json_response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=json_query,
                    config=GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=stock_schema
                    )
                )

                # Parse and return the JSON data
                try:
                    return json.loads(json_response.text)
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, try to extract JSON from the text
                    try:
                        text_response = json_response.text
                        json_start = text_response.find('[')
                        json_end = text_response.rfind(']') + 1

                        if json_start >= 0 and json_end > json_start:
                            json_str = text_response[json_start:json_end]
                            return json.loads(json_str)
                        else:
                            # Try to find JSON with curly braces if square brackets aren't found
                            json_start = text_response.find('{')
                            json_end = text_response.rfind('}') + 1

                            if json_start >= 0 and json_end > json_start:
                                json_str = text_response[json_start:json_end]
                                # If it's a single object, wrap it in an array
                                return [json.loads(json_str)]
                            else:
                                return [{"ticker": "ERROR", "name": "Error parsing JSON", "rationale": text_response}]
                    except Exception:
                        return [{"ticker": "ERROR", "name": "Error parsing JSON", "rationale": json_response.text}]
            else:
                # If JSON is not requested, just print and return the search results
                print(search_results)
                return None

        except Exception as e:
            error_message = f"Error generating recommendations: {str(e)}"
            print(error_message)
            if return_json:
                return [{"ticker": "ERROR", "name": "Error generating recommendations", "rationale": str(e)}]
            else:
                print(error_message)
                return None

if __name__ == "__main__":
    recommender = TradingRecommender()
    # Test with a different sector
    results = recommender.get_recommendations("technology")

