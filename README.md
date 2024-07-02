# A simple Python package for the geodetector 

# Install

```
pip install py-geodetector
```

# Usage

A quick example of geodetector usage is given in the ./example.ipynb.

```python
from py_geodetector import load_example_data, GeoDetector

# load example data
df = load_example_data()

gd = GeoDetector(df)
# factor detect
factor_df = gd.factor_dector()

# interaction detect
interaction_df = gd.interaction_detector()
# or you can generate the interaction relationship as the same time
interaction_df, interaction_relationship_df = gd.interaction_detector(relationship=True)

# ecological detect
ecological_df = gd.ecological_detector()

# plot 
# use a heatmap visualize the interaction detect result, 
# red text means that the ecological detection results show a significant difference
gd.plot(value_fontsize=14, tick_fontsize=16, colorbar_fontsize=14);
```

# Reference

```
@article{wang2010geographical,
  title={Geographical detectors-based health risk assessment and its application in the neural tube defects study of the Heshun Region, China},
  author={Wang, Jin-Feng and Li, Xin-Hu and Christakos, George and Liao, Yi-Lan and Zhang, Tin and Gu, Xue and Zheng, Xiao-Ying},
  journal={International Journal of Geographical Information Science},
  volume={24},
  number={1},
  pages={107-127},
  year={2010},
  publisher={Taylor \& Francis}
}
```