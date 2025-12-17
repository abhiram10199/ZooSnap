def pred(file):

    print("Predict endpoint called")
    # print image
    with open(file, 'rb') as f:
        image_data = f.read()
    print(f"Image data length: {len(image_data)} bytes")

file = 'vsbg.jpg'  # Replace with your test file path
pred(file)  
