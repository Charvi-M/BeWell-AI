import requests
from bs4 import BeautifulSoup
import os

os.makedirs("data", exist_ok=True)



def therapy_who_guidelines():
    urls = [
        "https://www.who.int/teams/mental-health-and-substance-use/treatment-care",
        "https://www.who.int/teams/mental-health-and-substance-use/data-research",
        "https://www.who.int/teams/mental-health-and-substance-use/emergencies",
        "https://www.who.int/activities/improving-the-mental-and-brain-health-of-children-and-adolescents",
        "https://www.who.int/teams/mental-health-and-substance-use/treatment-care/innovations-in-psychological-interventions",
        "https://www.who.int/teams/mental-health-and-substance-use/treatment-care/who-caregivers-skills-training-for-families-of-children-with-developmental-delays-and-disorders",
        "https://www.who.int/teams/mental-health-and-substance-use/treatment-care/mental-health-gap-action-programme"
    ]

    with open("data/therapy/who_psych_guidelines.txt", "w", encoding="utf-8") as f:
        for url in urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n")
            f.write(f"--- Content from: {url} ---\n")
            f.write(text + "\n\n")

def scrape_cleveland_data():
        urls = [
        "https://my.clevelandclinic.org/health/diseases/21544-schizoaffective-disorder",
        "https://my.clevelandclinic.org/health/diseases/9536-anxiety-disorders",
        "https://my.clevelandclinic.org/health/articles/autism",
        "https://my.clevelandclinic.org/health/diseases/4784-attention-deficithyperactivity-disorder-adhd",
        "https://my.clevelandclinic.org/health/diseases/4568-schizophrenia",
        "https://my.clevelandclinic.org/health/diseases/9599-delusional-disorder",
        "https://my.clevelandclinic.org/health/diseases/9294-bipolar-disorder",
        "https://my.clevelandclinic.org/health/diseases/17788-cyclothymia",
        "https://my.clevelandclinic.org/health/diseases/9290-depression",
        "https://my.clevelandclinic.org/health/diseases/9292-persistent-depressive-disorder-pdd",
        "https://my.clevelandclinic.org/health/diseases/23940-generalized-anxiety-disorder-gad",
        "https://my.clevelandclinic.org/health/diseases/22709-social-anxiety",
        "https://my.clevelandclinic.org/health/diseases/separation-anxiety-disorder",
        "https://my.clevelandclinic.org/health/diseases/4451-panic-attack-panic-disorder",
        "https://my.clevelandclinic.org/health/diseases/24757-phobias",
        "https://my.clevelandclinic.org/health/diseases/9490-ocd-obsessive-compulsive-disorder",
        "https://my.clevelandclinic.org/health/diseases/17682-hoarding-disorder",
        "https://my.clevelandclinic.org/health/diseases/9888-body-dysmorphic-disorder",
        "https://my.clevelandclinic.org/health/diseases/9545-post-traumatic-stress-disorder-ptsd",
        "https://my.clevelandclinic.org/health/diseases/21760-adjustment-disorder",
        "https://my.clevelandclinic.org/health/diseases/22706-dermatillomania-skin-picking",
        "https://my.clevelandclinic.org/health/diseases/9880-trichotillomania",
        "https://my.clevelandclinic.org/health/diseases/9792-dissociative-identity-disorder-multiple-personality-disorder",
        "https://my.clevelandclinic.org/health/diseases/9789-dissociative-amnesia",
        "https://my.clevelandclinic.org/health/diseases/9791-depersonalization-derealization-disorder",
        "https://my.clevelandclinic.org/health/diseases/17976-somatic-symptom-disorder-in-adults",
        "https://my.clevelandclinic.org/health/diseases/9886-illness-anxiety-disorder-hypochondria-hypochondriasis",
        "https://my.clevelandclinic.org/health/diseases/17975-conversion-disorder",
        "https://my.clevelandclinic.org/health/diseases/9794-anorexia-nervosa",
        "https://my.clevelandclinic.org/health/diseases/9795-bulimia-nervosa",
        "https://my.clevelandclinic.org/health/diseases/17652-binge-eating-disorder",
        "https://my.clevelandclinic.org/health/diseases/22944-pica",
        "https://my.clevelandclinic.org/health/diseases/12119-insomnia",
        "https://my.clevelandclinic.org/health/diseases/12147-narcolepsy",
        "https://my.clevelandclinic.org/health/diseases/8718-sleep-apnea",
        "https://my.clevelandclinic.org/health/diseases/9497-restless-legs-syndrome",
        "https://my.clevelandclinic.org/health/diseases/22690-sex-addiction-hypersexuality-and-compulsive-sexual-behavior",
        "https://my.clevelandclinic.org/health/diseases/9905-oppositional-defiant-disorder",
        "https://my.clevelandclinic.org/health/diseases/9657-antisocial-personality-disorder",
        "https://my.clevelandclinic.org/health/diseases/9878-kleptomania",
        "https://my.clevelandclinic.org/health/diseases/3909-alcoholism",
        "https://my.clevelandclinic.org/health/diseases/15742-inhalant-abuse",
        "https://my.clevelandclinic.org/health/diseases/15252-delirium",
        "https://my.clevelandclinic.org/health/diseases/9164-alzheimers-disease",
        "https://my.clevelandclinic.org/health/diseases/8525-parkinsons-disease-an-overview",
        "https://my.clevelandclinic.org/health/diseases/14369-huntingtons-disease",
        "https://my.clevelandclinic.org/health/diseases/8874-traumatic-brain-injury",
        "https://my.clevelandclinic.org/health/diseases/9762-borderline-personality-disorder-bpd",
        "https://my.clevelandclinic.org/health/diseases/9742-narcissistic-personality-disorder",
        "https://my.clevelandclinic.org/health/diseases/6125-tardive-dyskinesia",
        "https://my.clevelandclinic.org/health/diseases/22703-neuroleptic-malignant-syndrome"
    ]

        with open("data/therapy/cleaveland.txt", "w", encoding="utf-8") as f:
            for url in urls:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator="\n")
                f.write(f"--- Content from: {url} ---\n")
                f.write(text + "\n\n")


def scrape_who_guidelines():
    urls = [
        "https://www.who.int/teams/mental-health-and-substance-use/treatment-care",
        "https://www.who.int/teams/mental-health-and-substance-use/data-research",
        "https://www.who.int/teams/mental-health-and-substance-use/emergencies",
        "https://www.who.int/activities/improving-the-mental-and-brain-health-of-children-and-adolescents",
        "https://www.who.int/teams/mental-health-and-substance-use/treatment-care/innovations-in-psychological-interventions",
        "https://www.who.int/teams/mental-health-and-substance-use/treatment-care/who-caregivers-skills-training-for-families-of-children-with-developmental-delays-and-disorders",
        "https://www.who.int/teams/mental-health-and-substance-use/treatment-care/mental-health-gap-action-programme"
    ]

    with open("data/resources/who_psych_guidelines.txt", "w", encoding="utf-8") as f:
        for url in urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n")
            f.write(f"--- Content from: {url} ---\n")
            f.write(text + "\n\n")

def scrape_meditations():
    url = "https://www.gutenberg.org/cache/epub/2680/pg2680-images.html"  
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    content = soup.get_text(separator="\n")
    with open("data/therapy/meditations.txt", "w", encoding="utf-8") as f:
        f.write(content)
   

def scrape_global_helplines():
    url = "https://en.wikipedia.org/wiki/List_of_suicide_crisis_lines"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n")
    with open("data/resources/global_helplines.txt", "w", encoding="utf-8") as f:
        f.write(text)

def free_resources_india():
    url = "https://www.thelivelovelaughfoundation.org/find-help/helplines"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n")
    with open("data/resources/free_resources_india.txt", "w", encoding="utf-8") as f:
        f.write(text)

def scrape_india_free_services():
    telemanas_info = """
India’s Tele-MANAS mental health helpline (free, 24x7):
14416 or 1-800-891-4416
Multilingual support for stress, anxiety, and depression.
"""

    fallback_nimhans_link = "https://www.nimhans.ac.in/helpline-nimhans/"

    try:
        url = fallback_nimhans_link
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n")
        nimhans_content = f"NIMHANS Helpline:\n{text}"
    except Exception as e:
        print("Could not fetch NIMHANS due to SSL or connection issue. Using fallback link.")
        nimhans_content = f"NIMHANS Helpline:\n{fallback_nimhans_link}"

    with open("data/resources/india_helplines.txt", "w", encoding="utf-8") as f:
        f.write(telemanas_info.strip())
        f.write("\n\n")
        f.write(nimhans_content.strip())

if __name__ == "__main__":
    print("Scraping mental health content...")
   
    scrape_who_guidelines()
    therapy_who_guidelines()
    scrape_meditations()
    scrape_global_helplines()
    free_resources_india()
    scrape_india_free_services()
    scrape_cleveland_data()
    
    print("Done. Files saved to /data")
