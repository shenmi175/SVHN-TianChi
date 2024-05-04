# 物体检测的边界框增强
## 不同的注释格式
边界框是标记图像上物体的矩形。存在多种边界框注释格式。每种格式都使用其特定表示边界框坐标的方式。Albumentations支持四种格式：pascal_voc、albumentations、coco 和 yolo。

我们来看看这些格式中的每一种，以及它们是如何表示边界框的坐标的。

作为例子，我们将使用一个名为“上下文中的常见物体”的数据集中的图像。它包含一个标记猫的边界框。图像宽度为640像素，高度为480像素。边界框的宽度为322像素，高度为117像素。

边界框的角点（x, y）坐标如下：左上角为 (x_min, y_min) 或 (98px, 345px)，右上角为 (x_max, y_min) 或 (420px, 345px)，左下角为 (x_min, y_max) 或 (98px, 462px)，右下角为 (x_max, y_max) 或 (420px, 462px)。如你所见，边界框角点的坐标是相对于图像的左上角计算的，其坐标为 (0, 0)。

![An example image with a bounding box from the COCO dataset](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_example.jpg)

### pascal_voc
`pascal_voc`是Pascal VOC数据集使用的一种格式。边界框的坐标用四个像素值编码：[x_min, y_min, x_max, y_max]。x_min和y_min是边界框左上角的坐标，x_max和y_max是边界框右下角的坐标。

此格式中示例边界框的坐标为 [98, 345, 420, 462]。

### albumentations
`albumentations`与`pascal_voc`相似，因为它也使用四个值[x_min, y_min, x_max, y_max]来表示边界框。但与`pascal_voc`不同，`albumentations`使用归一化的值。为了归一化这些值，我们将x轴和y轴的像素坐标除以图像的宽度和高度。

此格式中示例边界框的坐标为 [98 / 640, 345 / 480, 420 / 640, 462 / 480]，即 [0.153125, 0.71875, 0.65625, 0.9625]。

Albumentations在内部使用此格式来处理和增强边界框。

### coco
`coco`是Common Objects in Context COCO数据集使用的格式。

在`coco`格式中，边界框由四个像素值[x_min, y_min, width, height]定义。这些值是边界框左上角的坐标以及边界框的宽度和高度。

此格式中示例边界框的坐标为 [98, 345, 322, 117]。

### yolo
在`yolo`格式中，边界框由四个值[x_center, y_center, width, height]表示。x_center和y_center是边界框中心的归一化坐标。为了使坐标归一化，我们取标记边界框中心的x和y的像素值，并将x的值除以图像宽度，将y的值除以图像高度。宽度和高度表示边界框的宽度和高度，也是归一化的。

此格式中示例边界框的坐标为[((420 + 98) / 2) / 640, ((462 + 345) / 2) / 480, 322 / 640, 117 / 480]，即[0.4046875, 0.840625, 0.503125, 0.24375]。

![How different formats represent coordinates of a bounding box](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_formats.jpg)


# 边界框增强
与图像和掩码增强一样，边界框增强的过程包括4个步骤。

1. 导入所需的库。
2. 定义一个增强管道。
3. 从磁盘读取图像和边界框。
4. 将图像和边界框传递给增强管道，并接收增强后的图像和框。

**注意**

Albumentation中的某些变换不支持边界框。如果尝试使用它们，您将收到一个异常。请参考这篇文章，以检查一个变换是否可以增强边界框。https://albumentations.ai/docs/getting_started/transforms_and_targets/

## 步骤1. 导入所需的库
```python
import albumentations as A
import cv2
```

## 步骤2. 定义一个增强管道
这里是一个处理边界框的最小声明增强管道的例子。

```python
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco'))
```

注意，与图像和掩码增强不同，`Compose`现在有一个额外的参数`bbox_params`。你需要将`A.BboxParams`的一个实例传递给那个参数。`A.BboxParams`指定了处理边界框的设置。`format`设置边界框坐标的格式。

它可以是`pascal_voc`、`albumentations`、`coco`或`yolo`。这个值是必需的，因为Albumentation需要知道边界框的坐标源格式，以正确应用增强。

除了格式，`A.BboxParams`还支持一些其他设置。

这里是一个展示`A.BboxParams`所有可用设置的`Compose`例子：

```python
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))
```



### min_area 和 min_visibility 参数

`min_area`和`min_visibility`参数控制Albumentations在边界框大小因增强而改变后应该如何处理这些边界框。如果你应用了空间增强，例如裁剪图像的一部分或调整图像大小，边界框的大小可能会改变。

`min_area`是以像素为单位的值。如果增强后的边界框的面积小于`min_area`，Albumentations将丢弃该框。因此，返回的增强后的边界框列表将不包含该边界框。

`min_visibility`是一个介于0和1之间的值。如果增强后的边界框面积与增强前的边界框面积的比率小于`min_visibility`，Albumentations将丢弃该框。所以，如果增强过程切除了大部分边界框，那么这个框将不会出现在返回的增强后边界框列表中。

以下是一个包含两个边界框的示例图像。边界框坐标使用`coco`格式声明。

![An example image with two bounding boxes](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_without_min_area_min_visibility_original.jpg)

首先，我们应用`CenterCrop`增强，而不声明`min_area`和`min_visibility`参数。增强后的图像包含两个边界框。

![An example image with two bounding boxes after applying augmentation](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_without_min_area_min_visibility_cropped.jpg)

接下来，我们应用相同的`CenterCrop`增强，但这次我们也使用了`min_area`参数。现在，增强后的图像只包含一个边界框，因为另一个边界框在增强后的面积小于`min_area`，所以Albumentations丢弃了那个边界框。

![An example image with one bounding box after applying augmentation with 'min_area'](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_with_min_area_cropped.jpg)

最后，我们应用带有`min_visibility`的`CenterCrop`增强。增强后，结果图像不包含任何边界框，因为所有边界框的可见性都低于`min_visibility`设置的阈值。

![An example image with zero bounding boxes after applying augmentation with 'min_visibility'](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_with_min_visibility_cropped.jpg)

### 边界框的类别标签
除了坐标之外，每个边界框还应该有一个相关联的类别标签，用以表示边界框内部的对象是什么。传递边界框的标签有两种方式。

假设你有一张包含三个对象的示例图像：狗、猫和运动球。这些对象的边界框坐标使用`coco`格式分别是 [23, 74, 295, 388]、[377, 294, 252, 161] 和 [333, 421, 49, 49]。

![An example image with 3 bounding boxes from the COCO dataset](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/multiple_bboxes.jpg)

**1. 你可以通过将标签作为额外值添加到坐标列表中，与边界框坐标一起传递标签。**
对于上面的图像，带有类别标签的边界框将变为 [23, 74, 295, 388, 'dog']、[377, 294, 252, 161, 'cat'] 和 [333, 421, 49, 49, 'sports ball']。

**NOTE:** 类别标签可以是任何类型：整数、字符串或任何其他Python数据类型。例如，作为类别标签的整数值将如下所示：[23, 74, 295, 388, 18]、[377, 294, 252, 161, 17] 和 [333, 421, 49, 49, 37]。

此外，你可以为每个边界框使用多个类别值，例如 [23, 74, 295, 388, 'dog', 'animal']、[377, 294, 252, 161, 'cat', 'animal'] 和 [333, 421, 49, 49, 'sports ball', 'item']。

**2. 你可以将边界框的标签作为一个单独的列表传递（首选方式）。**

例如，如果你有三个边界框如 [23, 74, 295, 388]、[377, 294, 252, 161] 和 [333, 421, 49, 49]，你可以创建一个包含类别标签的单独列表，如 ['cat', 'dog', 'sports ball'] 或 [18, 17, 37]。接下来，你将这个带有类别标签的列表作为单独的参数传递给转换函数。Albumentations需要知道所有这些带有类别标签的列表的名称，以便正确地将它们与增强后的边界框结合。然后，如果一个边界框因为增强后不再可见而被丢弃，Albumentations也将丢弃该框的类别标签。使用`label_fields`参数设置转换中将包含边界框标签描述的所有参数的名称（详见步骤4）。


## 步骤3. 从磁盘读取图像和边界框
从磁盘读取图像。

```python
image = cv2.imread("/path/to/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

边界框可以以不同的序列化格式存储在磁盘上：JSON、XML、YAML、CSV等。因此，读取边界框的代码取决于磁盘上数据的实际格式。

读取磁盘上的数据后，你需要为Albumentations准备边界框。

Albumentations期望边界框被表示为一个列表的列表。每个列表包含有关单个边界框的信息。边界框定义应至少包含代表该边界框坐标的四个元素。这四个值的实际含义取决于边界框的格式（可能是pascal_voc、albumentations、coco或yolo）。除了四个坐标，每个边界框的定义还可以包含一个或多个额外值。你可以使用这些额外的值来存储有关边界框中对象的额外信息，如对象内部的类别标签。在增强过程中，Albumentations不会处理这些额外的值。库将返回它们，连同增强后边界框的更新坐标。

## 步骤4. 将图像和边界框传递给增强管道并接收增强后的图像和框
如步骤2中讨论的那样，有两种方法可以将类别标签与边界框坐标一起传递：

**1. 将类别标签与坐标一起传递**

因此，如果你有如下所示的三个边界框的坐标：

```python
bboxes = [
    [23, 74, 295, 388],
    [377, 294, 252, 161],
    [333, 421, 49, 49],
]
```

你可以为每个边界框添加一个类别标签作为列表中四个坐标的额外元素。现在，带有边界框及其坐标的列表将如下所示：

```python
bboxes = [
    [23, 74, 295, 388, 'dog'],
    [377, 294, 252, 161, 'cat'],
    [333, 421, 49, 49, 'sports ball'],
]
```

或者每个边界框有多个标签：

```python
bboxes = [
    [23, 74, 295, 388, 'dog', 'animal'],
    [377, 294, 252, 161, 'cat', 'animal'],
    [333, 421, 49, 49, 'sports ball', 'item'],
]
```

你可以使用任何数据类型来声明类别标签。它可以是字符串、整数或任何其他Python数据类型。

接下来，你将图像和边界框传递给转换函数，并接收增强后的图像和边界框。

```python
transformed = transform(image=image, bboxes=bboxes)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
```

![Example input and output data for bounding boxes augmentation](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_augmentation_example.jpg)

**2. 将类别标签作为转换函数的单独参数传递（首选方式）**

假设你有三个边界框的坐标：

```python
bboxes = [
    [23, 74, 295, 388],
    [377, 294, 252, 161],
    [333, 421, 49, 49],
]
```

你可以创建一个单独的列表，包含这些边界框的类别标签：

```python
class_labels = ['cat', 'dog', 'parrot']
```

然后你将边界框和类别标签一起传递给转换。注意，要传递类别标签，你需要使用在第2步创建`Compose`实例时在`label_fields`中声明的参数名称。在我们的例子中，我们将参数的名称设置为`class_labels`。

```python
transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
transformed_class_labels = transformed['class_labels']
```
通过这种方式，如果边界框在增强过程中被删除，因为它们不再可见，那么对应的类别标签也将被删除，确保返回的数据一致性和完整性。

![Example input and output data for bounding boxes augmentation with a separate argument for class labels](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_augmentation_example_2.jpg)

请注意，`label_fields`期望一个列表，因此你可以设置包含边界框标签的多个字段。如果你这样声明`Compose`：

```python
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels', 'class_categories']))
```

你可以使用这些多个参数来传递关于类别标签的信息，如下所示：

```python
class_labels = ['cat', 'dog', 'parrot']
class_categories = ['animal', 'animal', 'item']

transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels, class_categories=class_categories)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
transformed_class_labels = transformed['class_labels']
transformed_class_categories = transformed['class_categories']
```

这种方法允许你灵活地管理多种类型的标签信息，增强图像处理的同时也能确保标签数据的准确传递和更新。

# 更多使用示例

1. [使用 Albumentations 增强物体检测任务中的边界框](https://albumentations.ai/docs/examples/example_bboxes/)
2. [如果需要保留所有边界框，如何使用 Albumentations 完成检测任务](https://albumentations.ai/docs/examples/example_bboxes2/)
3. [展示。在各种实际任务的不同图像集上展示酷炫的增强实例。](https://albumentations.ai/docs/examples/showcase/)
