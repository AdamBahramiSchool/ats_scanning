import pymupdf
from PIL import Image
doc=pymupdf.open("resume.pdf")
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

class pdfParser:
    def __init__(self,pdf_path):
        self.pdf_path=pdf_path

    def open_pdf(self):
        doc=pymupdf.open(self.pdf_path)
        return doc
    
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
    
    def instruction_based_model(self,resume_data):
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer=AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        # Create a text generation pipeline
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        prompt=f'''Heres my resume information:{resume_data}I want you to group job experiences and bullet points with their appropriate job title, while including the date. I also want you to pair project bullet points with their appropriate projects, title and headers. The whole point is to group sections of text with their appropriate headers.\
        after doing so, please give me the output of text areas with their appropriate section headers, titles and dates.
        '''
        response=generator(prompt, max_new_tokens=300,do_sample=True, temperature=0.7)
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
                    text_content_arr.append({f'{section['section']}':f'{line}'})
            return text_content_arr
        
                    
                    

pdfObject=pdfParser("resume.pdf")
text_content=pdfObject.get_text_content()
for section in text_content:
    print(section)
response=pdfObject.instruction_based_model(text_content)
print(response[0]["generated_text"])





   
  
    
    
    