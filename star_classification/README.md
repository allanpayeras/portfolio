# Stellar classification using machine learning models

Stars fascinate humanity since its beginning. It has played a very important role in our history, providing us with a cosmic calendar which proved itself crucial for activities such as agriculture.

Modern science has made significant discoveries about the nature of these celestial bodies. We now know that stars are mainly made of different elements clustered together in a hot plasma fueled by nuclear fusion reactions.

We also know that stars undergo different phases in its life as different element groups are fused together changing the state of balance between the gravitational pull and the outwards radiation pressure.

Very relevant to study the evolution of a star during its lifecycle is its classification in a system called *Yerkes Spectral Classification*, introduced by William W. Morgan, Philip C. Keenan and Edith Kellman of the Yerkes Observatory.
The different types are correlated with different phases of stellar evolution, hence its astrophysical importance.

In this project, the application of machine learning models for classification of stars was investigated utilizing data from the *Hipparcos Astronomical Catalogue*.

Here we provide on overview of the project and its main results.
Further detailed information can be found in the corresponding [Jupyter notebook](/star_classification/stars.ipynb).

## Data access

The data from the Hipparcos Astronomical Catalogue was retrieved from the [VizieR](https://vizier.cds.unistra.fr/) archive of the Strasbourg astronomical Data Center (CDS).
The python package [pyVO](https://pyvo.readthedocs.io/en/latest/#using-pyvo) was utilized as an interface to fetch the data from VizieR.

A detailed description of the data available in the Hipparcos Catalogue can be found in its [documentation](https://www.cosmos.esa.int/web/hipparcos/catalogues).
For this project with the objective of stellar classification, the following quantities were selected:

| Quantity         | Description                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| HIP ID           | Unique ID of the star in the Hipparcos Catalogue.                                                    |
| Parallax         | The apparent change in position of the star due to measurement at different points in Earth's orbit. |
| Visual magnitude | The apparent magnitude (brightness) of the star as observed from Earth.                            |
| Hp magnitude     | The apparent magnitude as measured in the Hipparcos system.                                          |
| $B-V$            | The color index of the star in the Johnson UBV system.                                               |
| $V-I$            | The color index of the star in the Cousins' system.                                                  |
| Spectral Type    | The spectral type of the star.                                                                       |

To ensure the quality of the data, we also filtered the results so that only observations with relative parallax error smaller than 0.2 milliarcsecs and uncertainty in $B-V$ smaller or equal to 0.05 mag were selected.

## Data cleaning

For the classification task, we are interested in the luminosity classes of the stars as defined by the Morgan-Keenan-Kellman (MKK) classification system.
The different classes are summarized in the following table.

| Class | Star type            |
|-------|----------------------|
| I     | Luminous supergiants |
| II    | Bright giants        |
| III   | Normal giants        |
| IV    | Subgiants            |
| V     | Main sequence        |
| sd    | Subdwarfs            |
| D     | White dwarfs         |

The spectral type provided in the Catalogue not only reports the MKK luminosity classification but also the Harvard spectral classification.
The latter consists in a letter (O,B,A,F,G,K, or M) followed by a number (from 0 to 9) which is based on the spectral lines related to the temperature (or color) of the star.

As a consequence, we performed a data cleaning procedure that extracts the MKK classification from the information provided in the spectral type field of the Catalogue.

## Exploratory data analysis

After verifying that the cleaned dataset did not contain missing values and that the distribution of the different quantities do not present outliers, we derived the absolute magnitudes of the stars from the available apparent magnitudes.

The apparent magnitude $m$ of a star quantifies its brightness as seen from Earth.
However, the more distant a star is the less bright it will appear in the sky.
To correct for this distance effect and fairly report the true brightness of a star the absolute magnitude $M$ is introduced, which measures the brightness as seen at a fixed distance of 10 parsecs from the star.

The absolute and apparent magnitude are related by the expression
$$
M = m - 5 ( \log_{10}(1000/p) - 1),
$$
where $p$ is the parallax of the star in milliarcsec.
The parallax is the apparent change in position of the star in the sky when it is measured at different points in the Earth's orbit around the Sun.
The larger the parallax of a star the closer it is to Earth.

The luminosity classification, as rather clear from its name, is correlated to the luminosity of a star.
However, it is derived from analyzing specific spectral lines of the stars which closely depends on its surface gravity.
The luminosity itself is the total radiation flux of the star which is related to its absolute magnitude.

Using our dataset, we plotted in the figure below the absolute magnitude as a function of the $B-V$ color index.
The latter is correlated with the temperature of the star as well as the Harvard spectral classification. This type of plot is know as the *Hertzsprung-Russell (HR) diagram*.

![hr_diagram](/star_classification/figures/hr_diagram.png)

The stars are distributed in different regions of the HR diagram.
The branch extending from the left to the lower right corresponds to the main sequence stars (type V).
The horizontal branch around magnitude zero that extends to the right is populated be red giants (type III).
Above these branches we find the bright giants (type II) and supergiants (type I).
In the bottom left of the diagram, the white dwarfs are displayed (type D).

During the lifecycle of a star, it goes through different phases migrating from a certain type to another.
The evolution and the stellar types in the lifecycle of a star is mainly determined by its mass.
For this reason, HR diagrams are very important to identify the current stage of stellar evolution as well as understand their astrophysical processes.
The regions with a larger population of stars correspond to the stages in their evolution that last a longer time.

In the plots below, we show the HR diagram separately for each stellar type as informed by the Hipparcos data.

![hr_diagram_per_type](/star_classification/figures/hr_diagram_types.png)

Note that there are some substantial overlap for some of the stellar types, specially type IV with types III and V.
Therefore, utilizing the magnitude and color data from the Hipparcos dataset might be intrinsically challenging for classification models.

We also observe that the number of stars of types III, IV and V is vastly greater than the others, as clearly evidenced in the plot below.
This imbalance in classes is potentially problematic for training the classification models, which will be addressed later.

![type_frequency](/star_classification/figures/type_frequency.png)

## Classification models and training

In this project, we have trained different classification models provided by the `scikit-learn` library with the aim of comparing their corresponding performances.
The following models were utilized:

* Logistic regression.
* Decision tree.
* Random forest.
* Gradient boosting.
* Support vector machine.
* Multi-layer perceptron network.

The default value for the hyperparameters of the models, as provided by the library, were utilized except for the multi-layer perceptron where we defined three hidden layers with 32, 64, and 32 neurons, respectively.

For training and evaluating the performance of the models, the dataset was divided into training and testing sets where 20% of the total was destined to the latter set.
Additionally, the training set was normalized using a gaussian-standard scaler.
The same normalization was subsequently applied to the testing set.

As noted previously, an imbalance between the different stellar classes is present in the data.
This situation is undesirable since it may introduce biases to the trained models.
To mitigate this problem, we attributed weights to the different classes inversely proportional to the frequency of the corresponding stellar types.

## Results and model evaluation