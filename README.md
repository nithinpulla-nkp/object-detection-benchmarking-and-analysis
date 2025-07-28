# object-detection-benchmarking-and-analysis

## 📁 Dataset Setup (Pascal VOC 2012)

1. **Download** the official Pascal VOC 2012 dataset:
   ```bash
   # If you're on macOS and don't have wget, first install it using Homebrew:
   # brew install wget
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
   ```

2. **Extract the Dataset**  
   After downloading, create a `data/` directory (if it doesn’t exist) and extract the contents into it using:
   ```bash
   mkdir -p data/
   tar -xvf VOCtrainval_11-May-2012.tar -C data/
   ```

3. **Verify Directory Structure**  
   Once extracted, your folder structure should look like this:
   ```
   data/
   └── VOCdevkit/
       └── VOC2012/
           ├── JPEGImages/      ← Contains image files
           ├── Annotations/     ← XML annotation files
           ├── ImageSets/       ← Text files defining train/val/test splits
           ├── SegmentationClass/
           └── ...
   ```

4. **Exclude Dataset from Git**  
   The dataset is large and should not be committed to version control. Add the following lines to your `.gitignore`:
   ```
   # .gitignore
   data/VOCdevkit/
   *.tar
   ```