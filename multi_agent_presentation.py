import os
import requests
from enum import Enum
from pprint import pformat
import xml.etree.ElementTree as ET
from xml.dom import minidom
from xml.sax.saxutils import escape

from dotenv import load_dotenv
from openai import AzureOpenAI
from colorama import Fore, Style

load_dotenv()


# Variables
MODEL = os.getenv("MODEL")


# Agent types
class Agent(Enum):
    CREATE_SLIDES = "Slide creator"
    ENRICH_SLIDES = "Slide enricher"
    CREATE_TITLE = "Title creator"
    CREATE_QUIZ = "Quiz creator"
    FIND_IMAGE = "Image finder"
    END = "End process"


# Return a shared LLM model
def get_shared_llm():
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version="2024-08-01-preview",
    )


# Get the initial state of the conversation
def get_initial_state(text: str) -> dict:
    return {
        "input_text": text,
        "slides": "",
        "enriched_slides": "",
        "title": "",
        "quiz": "",
        "image": ""
    }


# Check if the image corresponds to the content
def image_checker_agent(image_url: str, search_query: str) -> bool:
    system_prompt = f"""
You are an Image checker agent. Your task is to check if the found image corresponds to the search query.

Instructions:
1. Examine the supplied image and the search query.
2. Determine if the image is relevant to the query.
3. Determine if the image contains text.

Search query:
{search_query}

Note:
The image should not contain any text.

Output:
Return "True" if the image is somehow relevant to the query and does not contain text. Otherwise return "False".
""".strip()

    client = get_shared_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
        ]}
    ]

    try:
        response = client.chat.completions.create(
            messages=messages, model=MODEL)
    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
        return False

    return response.choices[0].message.content == "True"


# Find image based on the content of the presentation
def image_finder_agent(state: dict):
    if state["enriched_slides"] == "":
        return ""

    system_prompt = f"""
You are an Image finder agent. Your task is to find an image that is relevant to the content of the presentation.

Instructions:
1. Read the slides.
2. Create a Google search query based on the content of the presentation.

Output:
Return only the Google search query.
""".strip()

    client = get_shared_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["enriched_slides"]}
    ]

    response = client.chat.completions.create(messages=messages, model=MODEL)
    search_query = response.choices[0].message.content

    # Search for the image
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cx = os.getenv("GOOGLE_CX")
    search_url = f"""https://www.googleapis.com/customsearch/v1?key={
        google_api_key}&cx={google_cx}&q={search_query}&searchType=image"""

    response = requests.get(search_url)
    search_results = response.json()

    if "items" in search_results:
        for image in search_results["items"]:
            image_url = image["link"]

            # Escape the URL to avoid XML parsing issues
            escaped_url = escape(image_url)

            # Check if the image corresponds to the content
            image_ok = image_checker_agent(image_url, search_query)

            if image_ok:
                return f"<image>{escaped_url}</image>"
            else:
                print(
                    Fore.RED + "Image does not correspond to the content. Checking next image..." + Style.RESET_ALL)
    else:
        return ""


# Create a quiz based on the content of the presentation
def quiz_creator_agent(state: dict):
    if state["enriched_slides"] == "":
        return ""

    system_prompt = f"""
You are a Quiz creator agent. Your task is to create a quiz based on the content of the presentation. The quiz should test the knowledge and understanding of the audience regarding the main concepts and ideas presented in the slides.

Instructions:
1. Read the slides.
2. Create a quiz that includes questions related to the main concepts and ideas.
3. Ensure that the questions are clear, concise, and relevant to the presentation.
4. The quiz should have 3 questions.

Output format:
<quiz>
    <question>
        <question_text>Question 1</question_text>
        <answer>Answer 1</answer>
    </question>
    <question>
        <question_text>Question 2</question_text>
        <answer>Answer 2</answer>
    </question>
    <question>
        <question_text>Question 3</question_text>
        <answer>Answer 3</answer>
    </question>
</quiz>
""".strip()

    client = get_shared_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["enriched_slides"]}
    ]

    response = client.chat.completions.create(messages=messages, model=MODEL)

    return response.choices[0].message.content


# Create a suitable title for the presentation
def title_creator_agent(state: dict):
    if state["enriched_slides"] == "":
        return ""

    system_prompt = f"""
You are a Title creator agent. You will be provided with a Power Point presentation. Your job is to create a suitable title based on the slides. The title should be concise and capture the main idea of the slide.

Instructions:
1. Read the slides.
2. Create a title that summarizes the main idea of the presentation.

Output format:
<title>Title of the presentation</title>
""".strip()

    client = get_shared_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["enriched_slides"]}
    ]

    response = client.chat.completions.create(messages=messages, model=MODEL)

    return response.choices[0].message.content


# Enrich the content of the slides
def slide_enricher_agent(state: dict):
    enriched_slides = ET.Element("slides")
    slides = ET.fromstring(state["slides"])

    for i in range(len(slides)):
        print(Style.DIM + f"Slide {i + 1}/{len(slides)}" + Style.RESET_ALL)

        system_prompt = f"""
You are a Slide enricher agent. Your job is to create enriched content for Power Point slides. You will be provided with the title and the general content of the current slide. Your task is to enrich the content by adding more details.

The length of the enriched content should be 2-3 paragraphs. Each paragraph should contain 1-2 short sentences. The style should be journalistic.

Current state:
{pformat(state, indent=2)}

Instructions:
1. Read the title and content of the slide.
2. Enrich the content by adding more details.
3. Ensure that the content is relevant and adds value to the slide.
4. Ensure that the content is short and concise.

Output format:
<slide>
    <title>Title of the slide</title>
    <paragraph>First paragraph</paragraph>
    <paragraph>Second paragraph</paragraph>
    etc...
</slide>
""".strip()

        client = get_shared_llm()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Enrich the content of this slide: {
                ET.tostring(slides[i]).decode()}"
             }
        ]

        while True:
            try:
                response = client.chat.completions.create(
                    messages=messages, model=MODEL)

                enriched_slide = ET.fromstring(
                    response.choices[0].message.content)
                enriched_slides.append(enriched_slide)
                break
            except ET.ParseError:
                print(
                    Fore.RED + "An error occurred while parsing the XML response." + Style.RESET_ALL)

    return ET.tostring(enriched_slides).decode()


# Create a structure for the presentation
def slide_creator_agent(state: dict):
    system_prompt = f"""
You are a Slide creator agent. Your job is to create a bare bones list of Power Point slides based on the input text. This list of slides will be used as a structure for the presentation. The slides will be enriched with content later.

Current state:
{pformat(state, indent=2)}

Instructions:
1. Identify main concepts, sub-concepts, and their relationships.
2. Create a structure for a presentation based on the identified concepts.
3. Create a list of slides that should be included in the presentation.

Note:
You should restrict yourself to a maximum of 4 slides. Keep it simple and focused.

Output format:
<slides>
    <slide>
        <title>Title of the slide</title>
        <content>Content of the slide</content>
    </slide>
    <slide>
        <title>Title of the slide</title>
        <content>Content of the slide</content>
    </slide>
</slides>
""".strip()

    client = get_shared_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Create a list of slides based on the input text."}
    ]

    while True:
        response = client.chat.completions.create(
            messages=messages, model=MODEL)

        try:
            ET.fromstring(response.choices[0].message.content)
            return response.choices[0].message.content
        except ET.ParseError:
            print(
                Fore.RED + "An error occurred while parsing the XML response." + Style.RESET_ALL)


# Orchestrator agent to coordinate the actions of other agents
def orchestrator_agent(state: dict):
    system_prompt = f"""
You are an Orchestrator agent. Your task is to coordinate the actions of other agents in order to create high quality Power Point presentations from the given input text.

Instructions:
1. The presentation should have a clear structure.
2. The slides should be enriched with additional details.
3. The presentation should have a suitable title.
4. The presentation should include a relevant image.
5. The presentation should include a quiz.
6. Finish the process when all the components are ready.

Current state:
{pformat(state, indent=2)}

Available agents:
- Slide creator: Create a clear structure for the presentation based on the input text.
- Slide enricher: Create enriched content for each slide.
- Title creator: Create a suitable title for the presentation based on the enriched content.
- Image finder: Find an image that is relevant to the content of the presentation.
- Quiz creator: Create a quiz based on the content of the presentation.
- End process: Finish the process.

Note:
Always check the current state before selecting the next agent.

Output:
Return only the name of the agent you want to use next: "Slide creator", "Slide enricher", "Title creator", "Image finder", "Quiz creator", "End process".
""".strip()

    client = get_shared_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Decide which agent to use next."}
    ]

    response = client.chat.completions.create(messages=messages, model=MODEL)

    return response.choices[0].message.content


# Create the presentation
def create_presentation(input_text: str):
    print(Fore.GREEN + "Creating presentation..." + Style.RESET_ALL)
    state = get_initial_state(input_text)
    memory = []

    while True:
        # Run the orchestrator agent
        print(Fore.GREEN + "Running Orchestrator agent..." + Style.RESET_ALL)
        next_agent = orchestrator_agent(state)
        print(Fore.CYAN + f"Next agent: {next_agent}" + Style.RESET_ALL)

        # End the process
        if next_agent == Agent.END.value:
            # Check if all components are ready
            if state["enriched_slides"] == "" or state["title"] == "" or state["image"] == "" or state["quiz"] == "":
                print(
                    Fore.RED + "The presentation is not complete. Please finish all components." + Style.RESET_ALL)
                if state["enriched_slides"] == "":
                    memory.append(
                        {"role": "user", "content": "The presentation is not complete. Please finish enriched slides."})
                elif state["title"] == "":
                    memory.append(
                        {"role": "user", "content": "The presentation is not complete. Please finish the title."})
                elif state["image"] == "":
                    memory.append(
                        {"role": "user", "content": "The presentation is not complete. Please find an image."})
                elif state["quiz"] == "":
                    memory.append(
                        {"role": "user", "content": "The presentation is not complete. Please create a quiz."})
                continue
            else:
                print(Fore.GREEN + "Presentation created." + Style.RESET_ALL)

                # Add title to the slides
                slides = ET.fromstring(state["enriched_slides"])
                title = ET.fromstring(state["title"])
                slides.insert(0, title)

                # Add image to the slides
                image = ET.fromstring(state["image"])
                slides.insert(1, image)

                # Add quiz to the slides
                quiz = ET.fromstring(state["quiz"])
                slides.append(quiz)

                return ET.tostring(slides).decode()

        # Run the selected agent
        try:
            if next_agent == Agent.CREATE_SLIDES.value:
                print(Fore.GREEN + "Running Slide creator..." + Style.RESET_ALL)
                response = slide_creator_agent(state)

                # Update state
                state["slides"] = response
                print(Style.DIM + "State updated" + Style.RESET_ALL)

            elif next_agent == Agent.ENRICH_SLIDES.value:
                print(Fore.GREEN + "Running Slide enricher..." + Style.RESET_ALL)
                response = slide_enricher_agent(state)

                # Update state
                state["enriched_slides"] = response
                print(Style.DIM + "State updated" + Style.RESET_ALL)

            elif next_agent == Agent.CREATE_QUIZ.value:
                print(Fore.GREEN + "Running Quiz creator..." + Style.RESET_ALL)
                response = quiz_creator_agent(state)

                if response == "":
                    print(
                        Fore.RED + "Quiz creator failed to generate a quiz." + Style.RESET_ALL)
                    response = "Quiz creator failed to generate a quiz. Please create enriched slides first."
                else:
                    # Update state
                    state["quiz"] = response
                    print(Style.DIM + "State updated" + Style.RESET_ALL)

            elif next_agent == Agent.CREATE_TITLE.value:
                print(Fore.GREEN + "Running Title creator..." + Style.RESET_ALL)
                response = title_creator_agent(state)

                if response == "":
                    print(
                        Fore.RED + "Title creator failed to generate a title." + Style.RESET_ALL)
                    response = "Title creator failed to generate a title. Please create enriched slides first."
                else:
                    # Update state
                    state["title"] = response
                    print(Style.DIM + "State updated" + Style.RESET_ALL)

            elif next_agent == Agent.FIND_IMAGE.value:
                print(Fore.GREEN + "Running Image finder..." + Style.RESET_ALL)
                response = image_finder_agent(state)

                if response == "":
                    print(
                        Fore.RED + "Image finder failed to find an image." + Style.RESET_ALL)
                    response = "Image finder failed to find an image. Please create enriched slides first."
                else:
                    # Update state
                    state["image"] = response
                    print(Style.DIM + "State updated" + Style.RESET_ALL)

            # Update memory
            memory.append({"role": "assistant", "content": response})
            print(Style.DIM + "Memory updated" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
            break


# Pretty print XML
def pretty_print_xml(xml_string: str):
    xml = minidom.parseString(xml_string)
    pretty_xml_as_string = xml.toprettyxml()
    print(pretty_xml_as_string)


# Convert XML to HTML
def xml_to_html(xml_string: str) -> str:
    root = ET.fromstring(xml_string)
    html = [
        "<html>",
        "<head>",
        "<link rel='stylesheet' type='text/css' href='styles.css'>",
        "</head>",
        "<body>",
        "<div class='slides'>"
    ]

    # Add title
    title = root.find('title').text
    html.append(f"<h1>{title}</h1>")

    # Add image
    image = root.find('image')
    html.append(f"<img src='{image.text}' />")

    # Add slides
    for slide in root.findall('slide'):
        html.append("<div class='slide'>")
        title = slide.find('title').text
        html.append(f"<h2>{title}</h2>")
        paragraphs = slide.findall('paragraph')

        for paragraph in paragraphs:
            html.append(f"<p>{paragraph.text}</p>")

        html.append("</div>")

    # Add quiz
    quiz = root.find('quiz')
    html.append("<div class='quiz'>")
    html.append("<h2>Quiz</h2>")
    questions = quiz.findall('question')
    for question in questions:
        html.append("<details class='question'>")
        question_text = question.find("question_text")
        html.append(f"""<summary><strong>{
                    question_text.text}</strong></summary>""")
        answer = question.find("answer")
        html.append(f"{answer.text}")
        html.append("</details>")
    html.append("</div>")

    html.extend(["</div>", "</body>", "</html>"])
    return "\n".join(html)


if __name__ == "__main__":
    caesar_example_text = """
Gaius Julius Caesar[a] (12 July 100 BC – 15 March 44 BC) was a Roman general and statesman. A member of the First Triumvirate, Caesar led the Roman armies in the Gallic Wars before defeating his political rival Pompey in a civil war, and subsequently became dictator from 49 BC until his assassination in 44 BC. He played a critical role in the events that led to the demise of the Roman Republic and the rise of the Roman Empire.

In 60 BC, Caesar, Crassus, and Pompey formed the First Triumvirate, an informal political alliance that dominated Roman politics for several years. Their attempts to amass political power were opposed by many in the Senate, among them Cato the Younger with the private support of Cicero. Caesar rose to become one of the most powerful politicians in the Roman Republic through a string of military victories in the Gallic Wars, completed by 51 BC, which greatly extended Roman territory. During this time he both invaded Britain and built a bridge across the river Rhine. These achievements and the support of his veteran army threatened to eclipse the standing of Pompey, who had realigned himself with the Senate after the death of Crassus in 53 BC. With the Gallic Wars concluded, the Senate ordered Caesar to step down from his military command and return to Rome. In 49 BC, Caesar openly defied the Senate's authority by crossing the Rubicon and marching towards Rome at the head of an army.[3] This began Caesar's civil war, which he won, leaving him in a position of near-unchallenged power and influence in 45 BC.

After assuming control of government, Caesar began a programme of social and governmental reform, including the creation of the Julian calendar. He gave citizenship to many residents of far regions of the Roman Republic. He initiated land reforms to support his veterans and initiated an enormous building programme. In early 44 BC, he was proclaimed "dictator for life" (dictator perpetuo). Fearful of his power and domination of the state, a group of senators led by Brutus and Cassius assassinated Caesar on the Ides of March (15 March) 44 BC. A new series of civil wars broke out and the constitutional government of the Republic was never fully restored. Caesar's great-nephew and adopted heir Octavian, later known as Augustus, rose to sole power after defeating his opponents in the last civil war of the Roman Republic. Octavian set about solidifying his power, and the era of the Roman Empire began.

Caesar was an accomplished author and historian as well as a statesman; much of his life is known from his own accounts of his military campaigns. Other contemporary sources include the letters and speeches of Cicero and the historical writings of Sallust. Later biographies of Caesar by Suetonius and Plutarch are also important sources. Caesar is considered by many historians to be one of the greatest military commanders in history.[4] His cognomen was subsequently adopted as a synonym for "Emperor"; the title "Caesar" was used throughout the Roman Empire, giving rise to modern descendants such as Kaiser and Tsar. He has frequently appeared in literary and artistic works.
        """.strip()

    churchill_example_text = """
    Sir Winston Leonard Spencer Churchill[a] (30 November 1874 – 24 January 1965) was a British statesman, military officer, and writer who was Prime Minister of the United Kingdom from 1940 to 1945 (during the Second World War) and again from 1951 to 1955. Apart from 1922 to 1924, he was a member of Parliament (MP) from 1900 to 1964 and represented a total of five constituencies. Ideologically an adherent to economic liberalism and imperialism, he was for most of his career a member of the Conservative Party, which he led from 1940 to 1955. He was a member of the Liberal Party from 1904 to 1924.

Of mixed English and American parentage, Churchill was born in Oxfordshire into the wealthy, aristocratic Spencer family. He joined the British Army in 1895 and saw action in British India, the Mahdist War and the Second Boer War, gaining fame as a war correspondent and writing books about his campaigns. Elected a Conservative MP in 1900, he defected to the Liberals in 1904. In H. H. Asquith's Liberal government, Churchill was president of the Board of Trade and Home Secretary, championing prison reform and workers' social security. As First Lord of the Admiralty during the First World War, he oversaw the Gallipoli campaign, but after it proved a disaster, was demoted to Chancellor of the Duchy of Lancaster. He resigned in November 1915 and joined the Royal Scots Fusiliers on the Western Front for six months. In 1917, he returned to government under David Lloyd George and served successively as Minister of Munitions, Secretary of State for War, Secretary of State for Air, and Secretary of State for the Colonies, overseeing the Anglo-Irish Treaty and British foreign policy in the Middle East. After two years out of Parliament, he was Chancellor of the Exchequer in Stanley Baldwin's Conservative government, returning sterling in 1925 to the gold standard, depressing the UK economy.

Out of government during his so-called "wilderness years" in the 1930s, Churchill took the lead in calling for rearmament to counter the threat of militarism in Nazi Germany. At the outbreak of the Second World War he was re-appointed First Lord of the Admiralty. In May 1940, he became prime minister, succeeding Neville Chamberlain. Churchill formed a national government and oversaw British involvement in the Allied war effort against the Axis powers, resulting in victory in 1945. After the Conservatives' defeat in the 1945 general election, he became Leader of the Opposition. Amid the developing Cold War with the Soviet Union, he publicly warned of an "iron curtain" of Soviet influence in Europe and promoted European unity. Between his terms, he wrote several books recounting his experience during the war. He was awarded the Nobel Prize in Literature in 1953. He lost the 1950 election but was returned to office in 1951. His second term was preoccupied with foreign affairs, especially Anglo-American relations and preservation of what remained of the British Empire, with India no longer a part of it. Domestically, his government's priority was their extensive housebuilding programme, in which they were successful. In declining health, Churchill resigned in 1955, remaining an MP until 1964. Upon his death in 1965, he was given a state funeral.

One of the 20th century's most significant figures, Churchill remains popular in the UK and the rest of the Anglosphere. He is generally viewed as a victorious wartime leader who played an integral role in defending liberal democracy against the spread of fascism. He has sometimes been criticised for his imperialism and certain comments on race, in addition to some wartime decisions such as area bombing, but historians nevertheless rank Churchill as one of the greatest British prime ministers.
    """.strip()

    presentation = create_presentation(caesar_example_text)
    # pretty_print_xml(presentation)

    html = xml_to_html(presentation)

    with open("presentation.html", "w") as file:
        file.write(html)

    print("Done!")
