import fitz
pdf = fitz.open("rostliny.pdf")

for page_num, page in enumerate(pdf, start=1):
    for img_index, img in enumerate(page.get_images()):
        xref = img[0]
        pix = fitz.Pixmap(pdf, xref)
        filename = f"image_page{page_num}_img{img_index}_xref{xref}.png"
        pix.save(filename)
        pix = None  # Free memory
