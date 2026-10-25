# SNIC

C 源码、许可证和编译脚本统一保存在 `seg2/common/_snic/`。Python ctypes 接口位于 `seg2/common/snic.py`。

在项目根目录编译：

```bash
bash seg2/common/_snic/build.sh
```

本地生成的 `libsnic.so` 不入库。默认从本目录加载，可用 `SEG2_SNIC_LIBRARY` 指向其他已编译的库。`build.sh` 支持从任意工作目录执行。

算法来源：[achanta/SNIC](https://github.com/achanta/SNIC)，许可证见 [LICENSE](LICENSE)。论文：R. Achanta and S. Süsstrunk, “Superpixels and Polygons Using Simple Non-Iterative Clustering”, CVPR 2017。
