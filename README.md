# invasive_plant_classifier

An invasive plant classifier trained on BingSearch Images scraped dataset with fastai. 
Gradio instance hosted on HuggingFace Space here: **https://huggingface.co/spaces/et-do/bc_invasive_plant_classifier**


## British Columbia Invasive Plants Identifier

In the main notebook, I aim to apply transfer learning methods to a PyTorch image classification CNN (resnet34) to be able to identify both the species and the level of invasiveness to British Columbia as deemed by https://www2.gov.bc.ca/gov/content/environment/plants-animals-ecosystems/invasive-species/priority-species/priority-plants

Currently, the BC government identifies invasive plants across 5 categories:

Prevent: Species determined to be high risk to BC and not yet established. Management objective is prevent the introduction and establishment.

Provincial EDRR: Species is high risk to B.C. and is new to the Province. Management objective is eradication.

Provincial Containment: Species is high risk with limited extent in B.C. but significant potential to spread. Management objective is to prevent further expansion into new areas with the ultimate goal of reducing the overall extent.

Regional containment/Control: Species is high risk and well established, or medium risk with high potential for spread. Management objective is to prevent further expansion into new areas within the region through establishment of containment lines and identification of occurrences outside the line to control.

Management: Species is more widespread but may be of concern in specific situations with certain high values - e.g., conservation lands, specific agriculture crops. Management objective is to reduce the invasive species impacts locally or regionally, where resources are available.

All of these categories could be extremely relevant to a free-to-use plant-indentifier web app. However, in the sake of API costs, resource management, and model complexity, the first version of the model will only be trained to recognize plants under the Provincial Containment category (n=6). As the web app won't be geographically restricted, being able to use it both inside BC and outside BC to identify these plants that have a management objective of limitting outer-provincial occurrences could provide immense value.

The notebook will walkthrough:
- Data gathering and validation
- Data preprocessing & augmentation via FastAi dataloaders
- Training the model on the new dataset, and using results to further clean the data
- Serving the model under a huggingface space
