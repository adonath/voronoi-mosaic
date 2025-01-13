# Image Mosaic Generator using Voronoi Tesellation

This repository implements a method to create image mosaics using Voronoi tessellation proposed by Yoshinori Dobashi and Toshiyuki Haga, Henry Johan,
and Tomoyuki Nishita (Eurographics Short Presentations, 2002, DOI: 10.2312/egs.20021036).

You can use it to create mosaics like this:


| Input Image | Voronoi Mosaic |
| -------- | ------- |
| ![Input Image](https://raw.githubusercontent.com/adonath/voronoi-mosaic/main/example-images/butterfly.jpg) | ![Voronoi Mosaic](https://raw.githubusercontent.com/adonath/voronoi-mosaic/main/example-images/butterfly-mosaic.jpg) |



## Usage

To run the script:
```bash
python voronoi_mosaic.py examples-images/butterfly.jpg --output-path examples-images/butterfly-mosaic.jpg
```

To get help on the parameters:
```bash
python voronoi_mosaic.py --help
```





