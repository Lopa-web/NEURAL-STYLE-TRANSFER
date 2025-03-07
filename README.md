# NEURAL-STYLE-TRANSFER

"COMPANY":CODTECH IT SOLUTIONS

"NAME":LOPAMUDRA SINGH

"INTERN ID":CODHC60

"DOMAIN":ARTIFICIAL INTELLIGENCE

"DURATION":8 WEEKS

"MENTOR":NEELA SANTOSH

"DESCRIPTION":This project implements Neural Style Transfer (NST) using PyTorch and a pre-trained VGG19 model. NST is a deep learning technique that applies the artistic style of one image (style image) to another image (content image) while preserving the original content structure.
Features:Uses VGG19’s feature maps to extract content and style representations.
Computes Content Loss by measuring the difference between the content image and the generated image.
Computes Style Loss using the Gram matrix to capture style patterns from the style image.
Optimizes an output image (initialized as the content image) using the LBFGS optimizer to match both the content and style constraints.
Iterative Image Generation with periodic visualization of the transformed image.
Final Output: The generated stylized image is saved as 'styled_image.jpg'.
Workflow:Load Content and Style Images: Convert images into tensors and normalize them.
Extract Features using the pre-trained VGG19 model.
Define Loss Functions:
Content Loss: Measures the mean squared error between content features.
Style Loss: Uses Gram matrices to capture style patterns.
Optimize the Target Image using LBFGS optimizer for iterative updates.
Display Intermediate Outputs during training.
Save Final Stylized Image as 'styled_image.jpg'.
Deliverable:Python script that allows users to apply artistic style transfer to any input images. Users can modify the content and style images to experiment with different artistic effects.

"OUTPUT":

![Image](https://github.com/user-attachments/assets/499f8e9c-8938-4b9b-9b33-d87b227e1233)
