# Baseline model attribution

The PatchTST supervised model, backbone and layers in `patchtst/` are from the authors’ public implementation, commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9`. These three files are distributed under the Apache License 2.0 in [patchtst/LICENSE](patchtst/LICENSE). We changed the import namespace to relative imports. [Upstream source](https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised)

The accompanying `RevIN.py` implements per-instance normalization and inverse normalization for this model. [RevIN reference](https://github.com/ts-kim/RevIN)

The two classes in `linear_cores.py` are independent, fixed-shape implementations of the study's MixLinear and MDLinear configurations. MixLinear uses a time-domain period grid and a low-frequency branch. MDLinear decomposes the history into trend and seasonal components. The authors’ reference versions used for comparison are [MixLinear at 42dbb98](https://github.com/aitianma/MixLinear/tree/42dbb98a5bbe64c13bc75b3cc07a9dc4acf20106) and [KDD Wind Power Forecast at 1e87c0e](https://github.com/shaido987/KDD_wind_power_forecast/tree/1e87c0e3f281a85664e125df487e5da2f6105297/methods/mdlinear). The public package includes its own implementations of these two cores.
