import pymupdf
from PIL import Image
doc=pymupdf.open("resume.pdf")
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json

import re
from huggingface_hub import login
import os 

class pdfParser:
    def __init__(self,pdf_path):
        self.pdf_path=pdf_path

    def open_pdf(self):
        doc=pymupdf.open(self.pdf_path)
        return doc
    
    def safe_parse_json(self, raw_text):
        try:
            # Remove markdown formatting and extra whitespace
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:-3].strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:-3].strip()
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print("❌ Failed to parse JSON:", e)
            print("Raw text was:\n", raw_text)
            return None
    
    def get_html(self):
        doc=self.open_pdf()
        page_html_arr=[]
        for page_index in range(len(doc)):
            page=doc[page_index]
            page_html=page.get_text('html')
            page_html_arr.append(page_html)
        return page_html_arr
        
    def get_images(self):
        doc=self.open_pdf()
        for page_index in range(len(doc)):
            page=doc['page_index']
            # extracting image data
            image_list=page.get_images()
            if image_list:
                print(f"Found Image at Page Index: {page_index}")
            else:
                print("No images on Resume")
            for image_index, image in enumerate(image_list, start=1):
                xref=image[0]
                pix = pymupdf.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                pix.save("page_%s-image_%s.png" % (page_index, image_index)) # save the image as png
                pix = None
    
    def get_table_data(self):
        doc=self.open_pdf()
        for page_index in range(len(doc)):
            page=doc[page_index]
        
        
        # extracting table data
        tabs=page.find_tables()
        print(f"Number of tables: {len(tabs.tables)} found on page page_index ")
        
        table_data=[]
        for tables in tabs.tables:
            print(tables.extract())
            table_data.append(tables.extract())
        return table_data
    
    
    def get_links(self):
        doc=self.open_pdf()
        for page_index in range(len(doc)):
            page=doc[page_index]
            link_data=[]
            # link extraction from pdf
            link=page.first_link
            while link:
                url=link.uri
                print(url)
                link_data.append(url)
                link = link.next
        return link_data
                
    def get_annotations(self):
        doc=self.open_pdf()
        for page_index in range(len(doc)):
            page=doc[page_index]
            annotations_arr=[]
            # collect annotations 
            for annotations in page.annots():
                print('Annotations: ', annotations.get_text())
                annotations_arr.append(annotations.get_text())
            return annotations_arr
    
    def get_widget_info(self):
        doc=self.open_pdf()
        for page_index in range(len(doc)):
            page=doc[page_index]
            #widgets
            widget_data=[]
            for field in page.widgets():
                print(field.get_text()) 
                widget_data.append(field.get_text())
        return widget_data
    
    def get_document_metadata(self):
        document_metadata=[]
        doc=self.open_pdf()
        print('Metadata: ', doc.metadata)
        document_metadata.append(doc.metadata)
        return document_metadata
    
    def page_content_raster_image(self):
        doc=self.open_pdf()
        for page_index in range(len(doc)):
            page=doc[page_index]
            # create raster image of page content
            pix=page.get_pixmap()
            pix.save('pagecontent_{page_index}.png')
        
    
    def huggingface_login(self):
        login(token=os.getenv("huggingface_token"))
        print("Login sucessful")
    
    def extract_job_keywords(self,job_description):
        genai.configure(api_key=os.getenv("google_api_key"))
        model=genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Given the following job description, extract and categorize the relevant information to help optimize a résumé for ATS (Applicant Tracking System) scanning.

        Specifically, identify and organize:
        - **Keywords** (e.g., specific technologies, programming languages, methodologies, certifications)
        - **Technical Skills / Hard Skills** (e.g., Python, Docker, REST APIs, Machine Learning)
        - **Soft Skills** (e.g., communication, problem-solving, teamwork)
        - Any other relevant categories (e.g., Tools & Platforms, Cloud Providers)

        Output the result in structured JSON like this:

        {{
        "Keywords": [
            "Python",
            "Java",
            "REST APIs",
            "Agile",
            ...
        ],
        "Hard Skills": [
            "Machine Learning",
            "Docker",
            ...
        ],
        "Soft Skills": [
            "Team Collaboration",
            "Communication",
            ...
        ],
        "Tools & Platforms": [
            "AWS",
            "PostgreSQL",
            ...
        ]
        }}

        Here is the job description to process:
        {job_description}
        """
        response = model.generate_content(prompt)
        return response
        
    def score_resume(self,structured_resume:str, needed_keywords:str):
        genai.configure(api_key=os.getenv("google_api_key"))
        model=genai.GenerativeModel("gemini-1.5-flash")
        
        structured_resume = self.safe_parse_json(structured_resume)
        needed_keywords = self.safe_parse_json(needed_keywords)
        if not structured_resume or not needed_keywords:
            return {'error' : 'Invalid JSON'}
        resume_text=json.dumps(structured_resume).lower()
        
        score_prompt = f"""
        Compare the résumé and job description below to compute an ATS relevance score from 0 to 100.

        Factors to consider:
        - How many important keywords and skills from the job are present in the résumé?
        - Are they mentioned in strong contexts (e.g., in EXPERIENCE or PROJECTS)?
        - Is the candidate missing any core qualifications?
        - Consider both exact matches and close semantic matches.

        Output JSON in this format:

        {{
        "ATS_Score": 85,
        "Missing_Skills": ["Kubernetes", "GCP"],
        "Matched_Skills": ["Python", "Docker", "Machine Learning"],
        "Comments": "Strong experience with ML and cloud tools, missing GCP and k8s."
        }}

        Résumé:
        {structured_resume}

        Job Keywords and Skills:
        {needed_keywords}
        """
        gemini_score = genai.GenerativeModel("gemini-1.5-flash").generate_content(score_prompt)
        return gemini_score

        
    
    def instruction_based_model(self, resume_data):
        genai.configure(api_key=os.getenv("google_api_key"))
        model=genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        You will receive unstructured résumé data.

        Your task is to:
        - Identify and extract each top-level section (e.g., EXPERIENCE, PROJECTS, EDUCATION, TECHNICAL SKILLS).
        - For each section, group relevant content together:
            - In EXPERIENCE and PROJECTS, associate bullet points with their corresponding titles and metadata (like company name, dates, or location).
            - In TECHNICAL SKILLS, group tools or languages into subcategories if possible.

        Output the result as structured JSON in the following format:

        {{
        "EXPERIENCE": [
            {{
            "title": "LLM Software Developer",
            "organization": "FindGrant",
            "location": "Toronto, ON",
            "date_range": "March 2025 – May 2025",
            "bullets": [
                "Built AI pipelines...",
                "Used LangChain with Zephyr..."
            ]
            }},
            ...
        ],
        "PROJECTS": [
            {{
            "title": "Lung Cancer CT Scan Classifier",
            "date_range": "January 2025 – April 2025",
            "bullets": [
                "Built a deep learning model...",
                "Engineered an image augmentation pipeline..."
            ]
            }},
            ...
        ],
        "EDUCATION": [
            {{
            "institution": "Simon Fraser University",
            "location": "Burnaby, BC",
            "degree": "B.Sc. in Computing Science",
            "date_range": "September 2021 – September 2026"
            }}
        ],
        "TECHNICAL SKILLS": {{
            "Languages": [...],
            "Frameworks & Libraries": [...],
            "Databases": [...],
            "Tools & Platforms": [...],
            "Cloud": [...],
            "Networking & Systems": [...]
        }}
        }}

        Now, here is the résumé data to process:
        {resume_data}
        """

        response = model.generate_content(prompt)
        return response

    def get_text_content(self):
        def is_header(text):
            return re.match(r"^[A-Z\s]{5,}$", text.strip()) is not None
        
        doc=self.open_pdf()
        
        for page_index in range(len(doc)):
            page=doc[page_index]
            # blocks formatted like this: (x0, y0, x1, y1, text, block_no, block_type)
            blocks = page.get_text("blocks")
            # Sort by vertical position (y0) to mimic top-to-bottom reading
            blocks = sorted(blocks, key=lambda b: b[1]) # sorting by vertical position. top goes first
            text_categories=[]
            temp_arr=[]
            prev_header_section=None
            isheader_count=0
            for b in blocks:
                text = b[4].strip()

                if is_header(text):
                    isheader_count+=1
                    # Before switching to the new header, save the previous one with its content
                    if prev_header_section is not None:
                        text_categories.append({
                            "section": prev_header_section,
                            "content": temp_arr.copy()
                        })
                        temp_arr = []

                    prev_header_section = text  # this holds the actual header text
                else:
                    if isheader_count==0:
                        continue
                    temp_arr.append(text)
            
            if prev_header_section and temp_arr:
                text_categories.append({
                    "section": prev_header_section,
                    "content": temp_arr.copy()
                })

                # Print results
            text_content_arr=[]
            for section in text_categories:
                # print(f"\n=== {section['section']} ===")
                for line in section['content']:
                    # print(line)
                    text_content_arr.append({f"{section['section']}": f"{line}"})
            return text_content_arr
        
                    
                    

pdfObject=pdfParser("resume.pdf")
# pdfObject.huggingface_login()
text_content=pdfObject.get_text_content()
# for section in text_content:
#     print(section)

response_text= pdfObject.instruction_based_model(text_content)
structured_resume=response_text.candidates[0].content.parts[0].text
with open("job_desc.txt", 'r') as f:
    job_description = f.read()
needed_keywords_response=pdfObject.extract_job_keywords(job_description)
needed_keywords=needed_keywords_response.candidates[0].content.parts[0].text
score_response=pdfObject.score_resume(structured_resume,needed_keywords)
score=score_response.candidates[0].content.parts[0].text
print(score)







   
  
    
    
    