# Reference product images

Put images under numeric SKU IDs. The folder name is the exact identifier used in the index and planogram.

```
data/references/
  101/front.jpg
  101/shelf_view.jpg
  102/front.jpg
```

JPEG and PNG are supported. Images should contain one cropped product. Preserve leading zeros consistently.
Use **Products** in the dashboard to register a name and upload images, or create folders yourself.

After adding products, stop monitoring and click **Rebuild reference index**. This uses the exported embedding
model locally, preserves every SKU present in this folder, and does not train YOLO. Copy your complete
Colab reference catalog here before the first local rebuild; otherwise only local references will be indexed.
Keep your independent recognition validation/test images outside this folder.

