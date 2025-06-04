from openai import OpenAI

class LLM_Connector:
    """
    A class to communicate with LLM using the OpenAI API based on the PlugIR paper's Table 18 prompt template.
    """
    # Class-level attributes for prompt components that are constant
    QUESTION_SYSTEM_MESSAGE_CoT = """
    You are a proficient question generator tasked with aiding in the retrieval of a target image. Your role is to generate questions about the target image of the description via leveraging three key information sources:
    [Retrieval Candidates]: These are captions of images which are the candidates of the retrieval task for the target image described in [Description]. [Description]: This is a concise explanation of the target image.
    [Dialogue]: Comprising question and answer pairs that seek additional details about the target image.
    You should craft a question that narrows down the options for the attributes of the target image through drawing the information from the retrieval candidates. The generated question about the target image must be clear, succinct, and concise. Also, the question should only be asked about common objects in the description and candidates, which cannot be answered only from the description and the dialogue.
    Please explain how did you utilize the information sources for generating a question.
    """

    QUESTION_TRAIN_EXAMPLE_USER_CoT = """
    [Retrieval Candidates]
    0. A man in a yellow shirt.
    1. A boy in a skateboard park.
    2. The biker is performing a trick.
    3. A man in a green hat doing half-pipe with a skateboard.
    4. A skateboarding man catches the air in the midst of a trick.
    [Description]
    A man is doing a trick on a skateboard.
    [Dialogue]
    Question: What type of trick is the man performing on the skateboard? Answer: a jump.
    Question: What is the location of the jump trick being performed? Answer: a skate park.
    Answer: a skate park Question:
    """

    QUESTION_TRAIN_EXAMPLE_ASSISTANT_CoT = """
    Question: What is the outfit of the man performing the jump trick at a skate park? 
    Explanation: To generate a question about the description, I will utilize the retrieval candidates that mention the outfit of the man. Candidates 0 and 3 provide information about the man's wearing. The description mentions the man's trick on a skateboard, and the dialogue mentions the type and location of the trick. Since the attribute about the outfit does not appear in the description and the dialogue, the generated question cannot be answered from the information in the description and the dialogue about the target image. Additionally, the generated question is asking for the common objective, man, in the descriptions and candidates, not for the different objective from the description and the retrieval candidates 0 and 3, for example, a shirt and a half-pipe.
    """

    QUESTION_SYSTEM_MESSAGE = """
    You are a proficient question generator tasked with aiding in the retrieval of a target image. Your role is to generate questions about the target image of the description via leveraging two key information sources:
    [Description]: This is a concise explanation of the target image. [Dialogue]: Comprising question and answer pairs that seek additional details about the target image. Your generated question about the description must be clear, succinct, and concise, while differing from prior questions in the [Dialogue].
    """

    QUESTION_TRAIN_EXAMPLE_USER = """
    [Description] a man is doing a trick on a skateboard
    [Dialogue] Question: What type of trick is the man performing on the skateboard? Answer: a jump
    Question: What is the location of the jump trick being performed? Answer: a skate park Question:
    """

    QUESTION_TRAIN_EXAMPLE_ASSISTANT = """
    what is the outfit of the man performing the jump trick at a skate park?
    """

    FILTER_SYSTEM_MESSAGE = """
    Answer the question only according to the given context. If you cannot determine the answer or there are no objects that are asked by the question in the context , answer "Uncertain".
    """

    QUESTION_SYSTEM_MESSAGE_RECONSTURCTOR = """
    Your role is to reconstruct the [Caption] with the additional information given by following [Dialogue].
    The reconstructed [New Caption] should be concise and in appropriate form to retrieve a target image from a pool of candidate images.
    """

    QUESTION_TRAIN_EXAMPLE_USER_RECONSTURCTOR = """
    [Caption]: a woman sits on a bench holding a guitar in her lap [Dialogue]: is this in a park? yes, i believe it is, are there others around? no, she is alone, does she have a collection bucket? no, is her hair long? yes, pretty long, is she wearing a dress? i don’t think so, hard to tell, does she have shoes on? yes, flip flops, is there grass nearby? yes, everywhere, is it a sunny day? yes, are there trees? in the background there are trees, is the guitar new? i don’t think so 
    [New Caption]:
    """

    QUESTION_TRAIN_EXAMPLE_ASSISTANT_RECONSTURCTOR = """
    a woman with pretty long hair sits alone on a grassy bench in a park on a sunny day, holding a guitar in her lap without a collection bucket, wearing flip flops, with trees in the background, with a slightly worn guitar
    """


    def __init__(self, api_key: str, base_url: str, model_name: str = "grok-3"):
        """
        Initializes the QuestionGenerator with OpenAI API credentials and model.

        Args:
            api_key (str): Your OpenAI API key.
            base_url (str): The base URL for the OpenAI API.
            model_name (str): The name of the model to use (e.g., "deepseek-chat", "gpt-3.5-turbo").
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate_question(self, retrieval_context: dict, description: str, dialogue: list, mode: str = 'baseline') -> str | None:
        """
        Generate a question using the OpenAI API.

        Args:
            retrieval_context (dict): Dictionary mapping image IDs to generated captions.
            description (str): Initial user description (D0).
            dialogue (list): List of (question, answer) tuples; empty for first turn.
            mode (str): select prompting template from baseline and Chain of Thought (CoT)
        Returns:
            str: The generated question, or None if an error occurs.
        """
        retrieval_candidates = list(retrieval_context.values())

        rc_text = "[Retrieval Candidates]\n" + "\n".join(f"{i+1}. {cap}" for i, cap in enumerate(retrieval_candidates))
        desc_text = "[Description]\n" + description
        dial_text = "[Dialogue]\n" + ("None" if not dialogue else "\n".join(f"Question: {q} Answer: {a}" for q, a in dialogue))
        actual_user_message = f"{rc_text}\n{desc_text}\n{dial_text}\nQuestion:"

        # print(f'actual_user_message: {actual_user_message}') # Optional: for debugging

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.QUESTION_SYSTEM_MESSAGE_CoT},
                    {"role": "user", "content": self.QUESTION_TRAIN_EXAMPLE_USER_CoT},
                    {"role": "assistant", "content": self.QUESTION_TRAIN_EXAMPLE_ASSISTANT_CoT},
                    {"role": "user", "content": actual_user_message}
                ],
                max_tokens=32, # Consider making this configurable if needed
                temperature=0.7,
                n=1
            )
            full_response = response.choices[0].message.content.strip()
            
            if "Question:" in full_response:
                question_part = full_response.split("Question:", 1)[1] # Split only on the first occurrence
                question = question_part.split("\n")[0].strip()
                return question
            # If "Question:" is not found, but there's a response, return the whole response.
            # This might indicate the model didn't follow the format perfectly.
            return full_response if full_response else None
        except Exception as e:
            print(f"Error generating question: {e}")
            return None
        

    def filter_question(self, description: str, dialogue: list) -> str | None:
            """
            FILTER a question using the OpenAI API.

            Args:
                retrieval_context (dict): Dictionary mapping image IDs to generated captions.
                description (str): Initial user description (D0).
                dialogue (list): List of (question, answer) tuples; empty for first turn.
                mode (str): select prompting template from baseline and Chain of Thought (CoT)
            Returns:
                str: The generated question, or None if an error occurs.
            """
        
            desc_text = "[Description]" + description
            dial_text = dialogue_former(dialogue)
            context = "[Context]\n" + desc_text + '\n' + dial_text
            actual_user_message = f"{context}\nAnswer:" 

            # print(f'actual_user_message: {actual_user_message}') # Optional: for debugging
            print(f'Actual_user_message: {actual_user_message}')
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.FILTER_SYSTEM_MESSAGE},            
                        {"role": "user", "content": actual_user_message}
                    ],
                    max_tokens=10, # Consider making this configurable if needed
                    temperature=0,
                    n=1
                )
                full_response = response.choices[0].message.content.strip()
                
                # if "Question:" in full_response:
                #     question_part = full_response.split("Question:", 1)[1] # Split only on the first occurrence
                #     question = question_part.split("\n")[0].strip()
                #     return question
                # If "Question:" is not found, but there's a response, return the whole response.
                # This might indicate the model didn't follow the format perfectly.
                return full_response if full_response else None
            except Exception as e:
                print(f"Error generating question: {e}")
                return None


    def reformulate(self, description: str, dialogue: list) -> str | None:
        """
        reformulate questions using the OpenAI API.

        Args:
            retrieval_context (dict): Dictionary mapping image IDs to generated captions.
            description (str): Initial user description (D0).
            dialogue (list): List of (question, answer) tuples; empty for first turn.
            mode (str): select prompting template from baseline and Chain of Thought (CoT)
        Returns:
            str: The generated question, or None if an error occurs.
        """
        desc_text = "[Caption]: " + description
        dial_text = dialogue_former(dialogue)
        actual_user_message = f"{desc_text}\n{dial_text}\n[New Caption]:"

        print(f'actual_user_message: {actual_user_message}') # Optional: for debugging

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.QUESTION_SYSTEM_MESSAGE_RECONSTURCTOR},
                    {"role": "user", "content": self.QUESTION_TRAIN_EXAMPLE_USER_RECONSTURCTOR},
                    {"role": "assistant", "content": self.QUESTION_TRAIN_EXAMPLE_ASSISTANT_RECONSTURCTOR},
                    {"role": "user", "content": actual_user_message}
                ],
                max_tokens=32, # Consider making this configurable if needed
                temperature=0.7,
                n=1
            )
            full_response = response.choices[0].message.content.strip()
            print(f'LLM reformulat full_response: {full_response}')
            if "Question:" in full_response:
                question_part = full_response.split("Question:", 1)[1] # Split only on the first occurrence
                question = question_part.split("\n")[0].strip()
                return question
            # If "Question:" is not found, but there's a response, return the whole response.
            # This might indicate the model didn't follow the format perfectly.
            return full_response if full_response else None
        except Exception as e:
            print(f"Error generating question: {e}")
            return None
        

def dialogue_former(dialogue: list):
    dialogue_parts = []
    for item in dialogue:
        question = item['Question']
        answer = item['Answer']
        dialogue_parts.append(f"Question: {question} Answer: {answer}")
    dialogue_form = "[Dialogue] " + " ".join(dialogue_parts)
    return dialogue_form

# Example usage (if this file is run directly):
if __name__ == "__main__":
    # Replace with your actual API key and base URL
    API_KEY = ""
    BASE_URL = ""

    # Initialize the QuestionGenerator
    generator = LLM_Connector(api_key=API_KEY, base_url=BASE_URL)

    retrieval_context_example = {
        "unlabeled2017/000000000024.jpg": "a brick building",
        "unlabeled2017/000000000097.jpg": "a group of people walking down a street next to a river",
        "unlabeled2017/000000000207.jpg": "a person skateboarding in a park"
    }
  #  description_example = "a man on a skateboard"
    dialogue_example = []  # First turn, no dialogue

    # print("--- Example 1: First Turn ---")
    # question1 = generator.generate_question(retrieval_context_example, description_example, dialogue_example)
    # if question1:
    #     print("Generated Question:", question1)
    # else:
    #     print("Failed to generate question.")


    description_example = "man walking"
    context = [{'Question': 'Is the man walking indoors or outdoors?', 'Answer': 'the man is walking outdoors'}, {'Question': 'is the man holding a stick?', 'Answer': 'the man is holding a stick'}]
    # filter1 = generator.filter_question(description_example, context)
    reformulate = generator.reformulate(description_example, context)
    print(reformulate)


