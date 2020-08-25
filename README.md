## Prediction of the suspended particle matter in the Wadden Sea using machine learning and remote sensing

For the folks who don't know where the Wadden Sea is, it is located in the noth-west part of the Netherlands, as sown in the figure of below:

![alt text](https://github.com/anamabo/waddenAI/blob/master/plots/wadden.PNG)


Suspended Particulate Matter (SPM) in water is composed of particles that are larger than 2 microns (See Figure 1).  SPM is largely composed of inorganic material that vary in size and density but also consists of organic material, such as algae and bacteria. The variation in the distribution of inorganic material in the SPM makes resuspension processes hard to simulate, since different solids, located in the same water body, are resuspended at different time scales. SPM dynamics is also difficult to model numerically, since suspension depends on many physical, chemical and biological processes that are interconnected at different temporal frequencies.  In order to improve estimates of suspended matter in water systems we propose the use of data-driven models.


![alt text](https://github.com/anamabo/waddenAI/blob/master/plots/sediment.PNG)
Figure 1: Types of suspended matter



In the file called: phase1.pdf my team at Deltares and I built a data-driven model to predict SPM in the Dutch Wadden Sea. Both system dynamics (e.g. surge, wind velocity, and wind direction), as well as ecological and spatial-temporal effects (e.g. dissolved concentration of several chemical elements; chlorophyll-a content; months; seasons and tidal inlets) were used as forcing parameters. This data-driven model is able to make instant predictions of the SPM content in the Dutch Wadden Sea, considering all the complex processes that are present in the data. However, the model was trained on sparsely distributed, biweekly obtained, in-situ measurements, which makes it difficult to validate against physically-based numerical models.

Remote sensing might be a good solution to improve the spatial coverage of spm. In the second phase of the project, we explored this idea. In this repository you find the following material:

* Code/: The folder containing the Code to preprocess the data used to create the machibe learning model.
* project_all.pptx: Final presentation of the project.

The data to tranin the machine learning models is available upon request. 


