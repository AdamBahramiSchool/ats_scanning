import pymupdf
from PIL import Image
doc=pymupdf.open("resume.pdf")
output=open("output.txt", "wb") #with write back settings

for page_index in range(len(doc)):
    page=doc[page_index]
    text=page.get_text().encode('utf8')
    print(text)
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
    # extracting table data
    tabs=page.find_tables()
    print(f"Number of tables: {len(tabs.tables)} found on page page_index ")
    
    table_data=[]
    for tables in tabs.tables:
        print(tables.extract())
        table_data.append(tables.extract())
    
    link_data=[]
    # link extraction from pdf
    link=page.first_link
    while link:
        url=link.uri
        print(url)
        link_data.append(url)
        link = link.next
    annotations=[]
    # collect annotations 
    for annotations in page.annots():
        print('Annotations: ', annotations.get_text())
        annotations.append(annotations.get_text())
    document_metadata=[]
    print('Metadata: ', doc.metadata)
    document_metadata.append(doc.metadata)
    
    #widgets
    widget_data=[]
    for field in page.widgets():
        print(field.get_text()) 
        widget_data.append(field.get_text())
        
    # create raster image of page content
    pix=page.get_pixmap()
    pix.save('pagecontent.png')
    
    
        