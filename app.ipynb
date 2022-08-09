{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6848c5a6-85e6-41f5-9004-a690a77b5509",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastai.vision.all import *\n",
    "from fastai.vision.widgets import *\n",
    "import gradio as gr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f2fd99bd-54cd-42aa-bdbf-74cd795342f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "learn = load_learner('invsv_export.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a879266a-83d2-489d-aec1-2b558d8b5cb1",
   "metadata": {},
   "outputs": [],
   "source": [
    "labels = learn.dls.vocab\n",
    "def predict(img):\n",
    "    img = PILImage.create(img)\n",
    "    pred,pred_idx,probs = learn.predict(img)\n",
    "    return {labels[i]: float(probs[i]) for i in range(len(labels))}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dddb3a7e-2d8e-4392-a290-baf9fc5fb8e5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[0;31mType:\u001b[0m        list\n",
       "\u001b[0;31mString form:\u001b[0m ['examples/Alliaria petiolata.jpg', 'examples/anthriscus sylvestris.jpg', 'examples/Chondrilla juncea.jpg', 'examples/Conium maculatum.jpg', 'examples/Heracleum mantegazzianum.jpg', 'examples/Pastinaca sativa.jpg']\n",
       "\u001b[0;31mLength:\u001b[0m      6\n",
       "\u001b[0;31mDocstring:\u001b[0m  \n",
       "Built-in mutable sequence.\n",
       "\n",
       "If no argument is given, the constructor creates a new empty list.\n",
       "The argument must be an iterable if specified.\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "??examples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "6e62c71e-30c5-4451-a9ab-447bec05bb91",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.9/dist-packages/gradio/inputs.py:256: UserWarning: Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components\n",
      "  warnings.warn(\n",
      "/usr/local/lib/python3.9/dist-packages/gradio/deprecation.py:40: UserWarning: `optional` parameter is deprecated, and it has no effect\n",
      "  warnings.warn(value)\n",
      "/usr/local/lib/python3.9/dist-packages/gradio/outputs.py:196: UserWarning: Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components\n",
      "  warnings.warn(\n",
      "/usr/local/lib/python3.9/dist-packages/gradio/deprecation.py:40: UserWarning: The 'type' parameter has been deprecated. Use the Number component instead.\n",
      "  warnings.warn(value)\n",
      "/usr/local/lib/python3.9/dist-packages/gradio/deprecation.py:40: UserWarning: `enable_queue` is deprecated in `Interface()`, please use it within `launch()` instead.\n",
      "  warnings.warn(value)\n"
     ]
    }
   ],
   "source": [
    "title = \"BC Gov - Invasive Plant Classifier (Class: Provincial Containment)\"\n",
    "description = \"An invasive plant classifier, able to identify invasive plants deemed for 'Provincial Containment' (n=6) by the Government of British Columbia. Model trained on BingSearch Images scraped dataset with fastai. <br />Example Images:  <br />(1) *Alliaria petiolata* - Garlic Mustard<br />(2) *Anthriscus sylvestris* - Cow Parsley <br />(3) *Chondrilla juncea* - Rush Skeletonweed <br />(4) *Conium maculatum* - Poison Hemlock <br />(5) *Heracleum mantegazzianum* - Giant Hogweed <br />(6) *Pastinaca sativa* - Wild Parsnip\"\n",
    "examples = ['examples/Alliaria petiolata.jpg', 'examples/anthriscus sylvestris.jpg', 'examples/Chondrilla juncea.jpg', 'examples/Conium maculatum.jpg', 'examples/Heracleum mantegazzianum.jpg', 'examples/Pastinaca sativa.jpg']\n",
    "enable_queue=True\n",
    "\n",
    "iface = gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=6),title=title,description=description,examples=examples,enable_queue=enable_queue)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "f8017243-6aeb-4dd8-821d-5c50eaad322d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running on local URL:  http://127.0.0.1:7864/\n",
      "Running on public URL: https://38910.gradio.app\n",
      "\n",
      "This share link expires in 72 hours. For free permanent hosting, check out Spaces: https://huggingface.co/spaces\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"https://38910.gradio.app\" width=\"900\" height=\"500\" allow=\"autoplay; camera; microphone;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "(<gradio.routes.App at 0x7fa413ba12b0>,\n",
       " 'http://127.0.0.1:7864/',\n",
       " 'https://38910.gradio.app')"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iface.launch(share=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "388b7e1d-237b-4570-861f-17b4f88828d2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
