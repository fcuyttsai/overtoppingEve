??9
?G?G
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
A
AddV2
x"T
y"T
z"T"
Ttype:
2	??
?
	ApplyAdam
var"T?	
m"T?	
v"T?
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T?" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
s
	AssignAdd
ref"T?

value"T

output_ref"T?" 
Ttype:
2	"
use_lockingbool( 
s
	AssignSub
ref"T?

value"T

output_ref"T?" 
Ttype:
2	"
use_lockingbool( 
?
BatchDatasetV2
input_dataset

batch_size	
drop_remainder


handle"
parallel_copybool( "
output_types
list(type)(0" 
output_shapeslist(shape)(0
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FlatMapDataset
input_dataset
other_arguments2
Targuments

handle"	
ffunc"

Targuments
list(type)("
output_types
list(type)(0" 
output_shapeslist(shape)(0
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype?
is_initialized
"
dtypetype?
?
IteratorGetNext
iterator

components2output_types"
output_types
list(type)(0" 
output_shapeslist(shape)(0?
C
IteratorToStringHandle
resource_handle
string_handle?
?

IteratorV2

handle"
shared_namestring"
	containerstring"
output_types
list(type)(0" 
output_shapeslist(shape)(0?
,
MakeIterator
dataset
iterator?
?

MapDataset
input_dataset
other_arguments2
Targuments

handle"	
ffunc"

Targuments
list(type)("
output_types
list(type)(0" 
output_shapeslist(shape)(0"$
use_inter_op_parallelismbool(" 
preserve_cardinalitybool( 
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
PyFunc
input2Tin
output2Tout"
tokenstring"
Tin
list(type)("
Tout
list(type)(?
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
\
	RefSwitch
data"T?
pred

output_false"T?
output_true"T?"	
Ttype?
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
{
RepeatDataset
input_dataset	
count	

handle"
output_types
list(type)(0" 
output_shapeslist(shape)(0
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
;
	RsqrtGrad
y"T
dy"T
z"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
ShuffleDataset
input_dataset
buffer_size	
seed		
seed2	

handle"$
reshuffle_each_iterationbool("
output_types
list(type)(0" 
output_shapeslist(shape)(0
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
1
Square
x"T
y"T"
Ttype:

2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
?
StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
?
TensorSliceDataset

components2Toutput_types

handle"
Toutput_types
list(type)(0" 
output_shapeslist(shape)(0?
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.15.02v1.15.0-rc3-22-g590d6eef7e8Ȑ1
R
InputDatabasePlaceholder"/device:CPU:0*
dtype0*
_output_shapes
:
M
rndvaluePlaceholder"/device:CPU:0*
dtype0	*
_output_shapes
:
P
PlaceholderPlaceholder"/device:CPU:0*
_output_shapes
:*
dtype0
I
keepPlaceholder"/device:CPU:0*
_output_shapes
:*
dtype0
S
Training_forBNPlaceholder"/device:CPU:0*
dtype0
*
_output_shapes
:
T
accuracy_check_Placeholder"/device:CPU:0*
dtype0*
_output_shapes
:
v
flat_filenames/shapeConst"/device:CPU:0*
_output_shapes
:*
valueB:
?????????*
dtype0
{
flat_filenamesReshapeInputDatabaseflat_filenames/shape"/device:CPU:0*
T0*#
_output_shapes
:?????????
?
TensorSliceDatasetTensorSliceDatasetflat_filenames"/device:CPU:0*
_output_shapes
: *
Toutput_types
2*
output_shapes
: 
?
FlatMapDatasetFlatMapDatasetTensorSliceDataset"/device:CPU:0*
_output_shapes
: *
output_types
2*6
f1R/
-__inference_Dataset_flat_map_read_one_file_64*
output_shapes
: *

Targuments
 
?

MapDataset
MapDatasetFlatMapDataset"/device:CPU:0*

Targuments
 *
output_types
2*3
f.R,
*__inference_Dataset_map_decode_parse_fn_79*
_output_shapes
: *3
output_shapes"
 :?????????:?????????: 
U
seedConst"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R 
V
seed2Const"/device:CPU:0*
_output_shapes
: *
value	B	 R *
dtype0	
?
ShuffleDatasetShuffleDataset
MapDatasetrndvalueseedseed2"/device:CPU:0*
output_types
2*
_output_shapes
: *3
output_shapes"
 :?????????:?????????: 
V
countConst"/device:CPU:0*
_output_shapes
: *
dtype0	*
value	B	 R
?
RepeatDatasetRepeatDatasetShuffleDatasetcount"/device:CPU:0*3
output_shapes"
 :?????????:?????????: *
_output_shapes
: *
output_types
2
\

batch_sizeConst"/device:CPU:0*
dtype0	*
value
B	 R?*
_output_shapes
: 
_
drop_remainderConst"/device:CPU:0*
_output_shapes
: *
value	B
 Z *
dtype0

?
BatchDatasetV2BatchDatasetV2RepeatDataset
batch_sizedrop_remainder"/device:CPU:0*
_output_shapes
: *
output_types
2*Z
output_shapesI
G:??????????????????:??????????????????:?????????
?

IteratorV2
IteratorV2"/device:CPU:0*
_output_shapes
: *Z
output_shapesI
G:??????????????????:??????????????????:?????????*
output_types
2*
shared_name *
	container 
c
IteratorToStringHandleIteratorToStringHandle
IteratorV2"/device:CPU:0*
_output_shapes
: 
?
IteratorGetNextIteratorGetNext
IteratorV2"/device:CPU:0*[
_output_shapesI
G:??????????????????:??????????????????:?????????*
output_types
2*Z
output_shapesI
G:??????????????????:??????????????????:?????????
f
dataset_initMakeIteratorBatchDatasetV2
IteratorV2"/device:CPU:0*
_class
loc:@IteratorV2
s
strided_slice/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB"       
u
strided_slice/stack_1Const"/device:CPU:0*
valueB"       *
dtype0*
_output_shapes
:
u
strided_slice/stack_2Const"/device:CPU:0*
valueB"      *
_output_shapes
:*
dtype0
?
strided_sliceStridedSliceIteratorGetNextstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2"/device:CPU:0*#
_output_shapes
:?????????*
end_mask*

begin_mask*
Index0*
T0*
shrink_axis_mask
u
strided_slice_1/stackConst"/device:CPU:0*
dtype0*
valueB"        *
_output_shapes
:
w
strided_slice_1/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB"       
w
strided_slice_1/stack_2Const"/device:CPU:0*
valueB"      *
dtype0*
_output_shapes
:
?
strided_slice_1StridedSliceIteratorGetNext:1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2"/device:CPU:0*
Index0*
end_mask*
shrink_axis_mask*
T0*#
_output_shapes
:?????????*

begin_mask
g
MulMulstrided_slicestrided_slice_1"/device:CPU:0*
T0*#
_output_shapes
:?????????
u
strided_slice_2/stackConst"/device:CPU:0*
_output_shapes
:*
valueB"       *
dtype0
w
strided_slice_2/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB"       
w
strided_slice_2/stack_2Const"/device:CPU:0*
valueB"      *
_output_shapes
:*
dtype0
?
strided_slice_2StridedSliceIteratorGetNextstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2"/device:CPU:0*
T0*

begin_mask*#
_output_shapes
:?????????*
end_mask*
shrink_axis_mask*
Index0
e
truedivRealDivstrided_slice_2Mul"/device:CPU:0*
T0*#
_output_shapes
:?????????
x
transpose/aPacktruedivtruedivtruediv"/device:CPU:0*
N*'
_output_shapes
:?????????*
T0
n
transpose/permConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB"       
t
	transpose	Transposetranspose/atranspose/perm"/device:CPU:0*'
_output_shapes
:?????????*
T0
u
strided_slice_3/stackConst"/device:CPU:0*
_output_shapes
:*
valueB"       *
dtype0
w
strided_slice_3/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB"       
w
strided_slice_3/stack_2Const"/device:CPU:0*
valueB"      *
dtype0*
_output_shapes
:
?
strided_slice_3StridedSliceIteratorGetNext:1strided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2"/device:CPU:0*#
_output_shapes
:?????????*
shrink_axis_mask*
Index0*
end_mask*

begin_mask*
T0
u
strided_slice_4/stackConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB"       
w
strided_slice_4/stack_1Const"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB"       
w
strided_slice_4/stack_2Const"/device:CPU:0*
dtype0*
valueB"      *
_output_shapes
:
?
strided_slice_4StridedSliceIteratorGetNext:1strided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_2"/device:CPU:0*
end_mask*#
_output_shapes
:?????????*

begin_mask*
Index0*
shrink_axis_mask*
T0
i
Mul_1Mulstrided_slice_4	transpose"/device:CPU:0*'
_output_shapes
:?????????*
T0
u
strided_slice_5/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB"       
w
strided_slice_5/stack_1Const"/device:CPU:0*
valueB"       *
dtype0*
_output_shapes
:
w
strided_slice_5/stack_2Const"/device:CPU:0*
_output_shapes
:*
valueB"      *
dtype0
?
strided_slice_5StridedSliceIteratorGetNextstrided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_2"/device:CPU:0*
shrink_axis_mask*
Index0*

begin_mask*
T0*
end_mask*#
_output_shapes
:?????????
M
SizeSizestrided_slice_5"/device:CPU:0*
_output_shapes
: *
T0
u
strided_slice_6/stackConst"/device:CPU:0*
_output_shapes
:*
valueB"        *
dtype0
w
strided_slice_6/stack_1Const"/device:CPU:0*
_output_shapes
:*
valueB"       *
dtype0
w
strided_slice_6/stack_2Const"/device:CPU:0*
valueB"      *
dtype0*
_output_shapes
:
?
strided_slice_6StridedSliceIteratorGetNextstrided_slice_6/stackstrided_slice_6/stack_1strided_slice_6/stack_2"/device:CPU:0*
end_mask*
Index0*0
_output_shapes
:??????????????????*
T0*

begin_mask
`
Reshape/shape/1Const"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :
`
Reshape/shape/2Const"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0
z
Reshape/shapePackSizeReshape/shape/1Reshape/shape/2"/device:CPU:0*
N*
T0*
_output_shapes
:
w
ReshapeReshapestrided_slice_6Reshape/shape"/device:CPU:0*+
_output_shapes
:?????????*
T0
?
global_step/Initializer/ConstConst*
_output_shapes
: *
dtype0*
_class
loc:@global_step*
valueB
 *    
z
global_step
VariableV2"/device:CPU:0*
shape: *
_output_shapes
: *
_class
loc:@global_step*
dtype0
?
global_step/AssignAssignglobal_stepglobal_step/Initializer/Const"/device:CPU:0*
T0*
_output_shapes
: *
_class
loc:@global_step
y
global_step/readIdentityglobal_step"/device:CPU:0*
_class
loc:@global_step*
T0*
_output_shapes
: 
z
&ExponentialDecay/initial_learning_rateConst"/device:CPU:0*
valueB
 *??8*
_output_shapes
: *
dtype0
k
ExponentialDecay/Cast/xConst"/device:CPU:0*
valueB
 * ?z@*
_output_shapes
: *
dtype0
m
ExponentialDecay/Cast_1/xConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *)\?
{
ExponentialDecay/truedivRealDivPlaceholderExponentialDecay/Cast/x"/device:CPU:0*
T0*
_output_shapes
:
?
ExponentialDecay/PowPowExponentialDecay/Cast_1/xExponentialDecay/truediv"/device:CPU:0*
_output_shapes
:*
T0
?
ExponentialDecayMul&ExponentialDecay/initial_learning_rateExponentialDecay/Pow"/device:CPU:0*
_output_shapes
:*
T0
h
My_GPU_1/ExpandDims/dimConst"/device:GPU:1*
value	B :*
dtype0*
_output_shapes
: 
?
My_GPU_1/ExpandDims
ExpandDimsReshapeMy_GPU_1/ExpandDims/dim"/device:GPU:1*/
_output_shapes
:?????????*
T0
?
)My_GPU_1/CCN_1Conv_x0/strided_slice/stackConst"/device:GPU:1*
dtype0*
_output_shapes
:*!
valueB"            
?
+My_GPU_1/CCN_1Conv_x0/strided_slice/stack_1Const"/device:GPU:1*
dtype0*
_output_shapes
:*!
valueB"           
?
+My_GPU_1/CCN_1Conv_x0/strided_slice/stack_2Const"/device:GPU:1*!
valueB"         *
dtype0*
_output_shapes
:
?
#My_GPU_1/CCN_1Conv_x0/strided_sliceStridedSliceMy_GPU_1/ExpandDims)My_GPU_1/CCN_1Conv_x0/strided_slice/stack+My_GPU_1/CCN_1Conv_x0/strided_slice/stack_1+My_GPU_1/CCN_1Conv_x0/strided_slice/stack_2"/device:GPU:1*
Index0*
end_mask*

begin_mask*+
_output_shapes
:?????????*
T0*
shrink_axis_mask
?
<CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
dtype0*!
valueB"         *
_output_shapes
:
?
:CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
dtype0*
valueB
 *Iv??*
_output_shapes
: 
?
:CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
valueB
 *Iv?=*
dtype0
?
DCCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/RandomUniformRandomUniform<CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
dtype0*#
_output_shapes
:?
?
:CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/subSub:CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/max:CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
:CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/mulMulDCCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/RandomUniform:CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/sub*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?*
T0
?
6CCN_1Conv_x0/convA10/kernel/Initializer/random_uniformAdd:CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/mul:CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform/min*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?*
T0
?
CCN_1Conv_x0/convA10/kernel
VariableV2"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
dtype0*
shape:?*#
_output_shapes
:?
?
"CCN_1Conv_x0/convA10/kernel/AssignAssignCCN_1Conv_x0/convA10/kernel6CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?
?
 CCN_1Conv_x0/convA10/kernel/readIdentityCCN_1Conv_x0/convA10/kernel"/device:GPU:1*
T0*#
_output_shapes
:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
7My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/SquareSquare CCN_1Conv_x0/convA10/kernel/read"/device:GPU:1*#
_output_shapes
:?*
T0
?
6My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/ConstConst"/device:GPU:1*!
valueB"          *
_output_shapes
:*
dtype0
?
4My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/SumSum7My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Square6My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Const"/device:GPU:1*
T0*
_output_shapes
: 
?
6My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul/xConst"/device:GPU:1*
_output_shapes
: *
valueB
 *???3*
dtype0
?
4My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mulMul6My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul/x4My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Sum"/device:GPU:1*
T0*
_output_shapes
: 
?
6My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/add/xConst"/device:GPU:1*
valueB
 *    *
_output_shapes
: *
dtype0
?
4My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/addAddV26My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/add/x4My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul"/device:GPU:1*
T0*
_output_shapes
: 
?
+CCN_1Conv_x0/convA10/bias/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
dtype0
?
CCN_1Conv_x0/convA10/bias
VariableV2"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
shape:?*
dtype0
?
 CCN_1Conv_x0/convA10/bias/AssignAssignCCN_1Conv_x0/convA10/bias+CCN_1Conv_x0/convA10/bias/Initializer/zeros"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?
?
CCN_1Conv_x0/convA10/bias/readIdentityCCN_1Conv_x0/convA10/bias"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0
?
+My_GPU_1/CCN_1Conv_x0/convA10/dilation_rateConst"/device:GPU:1*
_output_shapes
:*
dtype0*
valueB:
?
3My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims/dimConst"/device:GPU:1*
dtype0*
value	B :*
_output_shapes
: 
?
/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims
ExpandDims#My_GPU_1/CCN_1Conv_x0/strided_slice3My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims/dim"/device:GPU:1*
T0*/
_output_shapes
:?????????
?
5My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims_1/dimConst"/device:GPU:1*
dtype0*
value	B : *
_output_shapes
: 
?
1My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims_1
ExpandDims CCN_1Conv_x0/convA10/kernel/read5My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims_1/dim"/device:GPU:1*'
_output_shapes
:?*
T0
?
$My_GPU_1/CCN_1Conv_x0/convA10/conv1dConv2D/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims_1"/device:GPU:1*
T0*
strides
*
paddingSAME*0
_output_shapes
:??????????
?
,My_GPU_1/CCN_1Conv_x0/convA10/conv1d/SqueezeSqueeze$My_GPU_1/CCN_1Conv_x0/convA10/conv1d"/device:GPU:1*
squeeze_dims
*,
_output_shapes
:??????????*
T0
?
%My_GPU_1/CCN_1Conv_x0/convA10/BiasAddBiasAdd,My_GPU_1/CCN_1Conv_x0/convA10/conv1d/SqueezeCCN_1Conv_x0/convA10/bias/read"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
"My_GPU_1/CCN_1Conv_x0/convA10/ReluRelu%My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
<CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*!
valueB"         *
dtype0*
_output_shapes
:
?
:CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *?5?*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel
?
:CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
dtype0*
valueB
 *?5=
?
DCCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/RandomUniformRandomUniform<CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/shape*
dtype0*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel
?
:CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/subSub:CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/max:CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/min*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
_output_shapes
: *
T0
?
:CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/mulMulDCCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/RandomUniform:CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/sub*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0*$
_output_shapes
:??
?
6CCN_1Conv_x0/convB10/kernel/Initializer/random_uniformAdd:CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/mul:CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform/min*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel
?
CCN_1Conv_x0/convB10/kernel
VariableV2"/device:GPU:1*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
shape:??*$
_output_shapes
:??
?
"CCN_1Conv_x0/convB10/kernel/AssignAssignCCN_1Conv_x0/convB10/kernel6CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel
?
 CCN_1Conv_x0/convB10/kernel/readIdentityCCN_1Conv_x0/convB10/kernel"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??
?
7My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/SquareSquare CCN_1Conv_x0/convB10/kernel/read"/device:GPU:1*$
_output_shapes
:??*
T0
?
6My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/ConstConst"/device:GPU:1*!
valueB"          *
_output_shapes
:*
dtype0
?
4My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/SumSum7My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Square6My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Const"/device:GPU:1*
T0*
_output_shapes
: 
?
6My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul/xConst"/device:GPU:1*
dtype0*
valueB
 *???3*
_output_shapes
: 
?
4My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mulMul6My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul/x4My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Sum"/device:GPU:1*
T0*
_output_shapes
: 
?
6My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/add/xConst"/device:GPU:1*
dtype0*
valueB
 *    *
_output_shapes
: 
?
4My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/addAddV26My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/add/x4My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul"/device:GPU:1*
_output_shapes
: *
T0
?
+CCN_1Conv_x0/convB10/bias/Initializer/zerosConst*
valueB?*    *
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
_output_shapes	
:?
?
CCN_1Conv_x0/convB10/bias
VariableV2"/device:GPU:1*
_output_shapes	
:?*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
shape:?
?
 CCN_1Conv_x0/convB10/bias/AssignAssignCCN_1Conv_x0/convB10/bias+CCN_1Conv_x0/convB10/bias/Initializer/zeros"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
?
CCN_1Conv_x0/convB10/bias/readIdentityCCN_1Conv_x0/convB10/bias"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
?
+My_GPU_1/CCN_1Conv_x0/convB10/dilation_rateConst"/device:GPU:1*
_output_shapes
:*
dtype0*
valueB:
?
3My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims/dimConst"/device:GPU:1*
value	B :*
dtype0*
_output_shapes
: 
?
/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims
ExpandDims"My_GPU_1/CCN_1Conv_x0/convA10/Relu3My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims/dim"/device:GPU:1*0
_output_shapes
:??????????*
T0
?
5My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_1/dimConst"/device:GPU:1*
dtype0*
value	B : *
_output_shapes
: 
?
1My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_1
ExpandDims CCN_1Conv_x0/convB10/kernel/read5My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_1/dim"/device:GPU:1*(
_output_shapes
:??*
T0
?
$My_GPU_1/CCN_1Conv_x0/convB10/conv1dConv2D/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_1"/device:GPU:1*
T0*0
_output_shapes
:??????????*
strides
*
paddingSAME
?
,My_GPU_1/CCN_1Conv_x0/convB10/conv1d/SqueezeSqueeze$My_GPU_1/CCN_1Conv_x0/convB10/conv1d"/device:GPU:1*
squeeze_dims
*
T0*,
_output_shapes
:??????????
?
%My_GPU_1/CCN_1Conv_x0/convB10/BiasAddBiasAdd,My_GPU_1/CCN_1Conv_x0/convB10/conv1d/SqueezeCCN_1Conv_x0/convB10/bias/read"/device:GPU:1*,
_output_shapes
:??????????*
T0
?
"My_GPU_1/CCN_1Conv_x0/convB10/ReluRelu%My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
+My_GPU_1/CCN_1Conv_x0/poolB10/dilation_rateConst"/device:GPU:1*
dtype0*
_output_shapes
:*
valueB:
}
,My_GPU_1/CCN_1Conv_x0/poolB10/ExpandDims/dimConst"/device:GPU:1*
dtype0*
_output_shapes
: *
value	B :
?
(My_GPU_1/CCN_1Conv_x0/poolB10/ExpandDims
ExpandDims"My_GPU_1/CCN_1Conv_x0/convB10/Relu,My_GPU_1/CCN_1Conv_x0/poolB10/ExpandDims/dim"/device:GPU:1*0
_output_shapes
:??????????*
T0
?
My_GPU_1/CCN_1Conv_x0/poolB10MaxPool(My_GPU_1/CCN_1Conv_x0/poolB10/ExpandDims"/device:GPU:1*
ksize
*0
_output_shapes
:??????????*
strides
*
paddingSAME
?
%My_GPU_1/CCN_1Conv_x0/poolB10/SqueezeSqueezeMy_GPU_1/CCN_1Conv_x0/poolB10"/device:GPU:1*
squeeze_dims
*,
_output_shapes
:??????????*
T0
?
<CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*!
valueB"         
?
:CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
dtype0*
valueB
 *?5?
?
:CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/maxConst*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
_output_shapes
: *
valueB
 *?5=
?
DCCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/RandomUniformRandomUniform<CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/shape*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0*
dtype0
?
:CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/subSub:CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/max:CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
:CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/mulMulDCCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/RandomUniform:CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/sub*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0
?
6CCN_1Conv_x0/convB20/kernel/Initializer/random_uniformAdd:CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/mul:CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform/min*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
CCN_1Conv_x0/convB20/kernel
VariableV2"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??*
shape:??*
dtype0
?
"CCN_1Conv_x0/convB20/kernel/AssignAssignCCN_1Conv_x0/convB20/kernel6CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
 CCN_1Conv_x0/convB20/kernel/readIdentityCCN_1Conv_x0/convB20/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0*$
_output_shapes
:??
?
7My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/SquareSquare CCN_1Conv_x0/convB20/kernel/read"/device:GPU:1*
T0*$
_output_shapes
:??
?
6My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/ConstConst"/device:GPU:1*
_output_shapes
:*!
valueB"          *
dtype0
?
4My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/SumSum7My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Square6My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Const"/device:GPU:1*
T0*
_output_shapes
: 
?
6My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul/xConst"/device:GPU:1*
valueB
 *???3*
dtype0*
_output_shapes
: 
?
4My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mulMul6My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul/x4My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Sum"/device:GPU:1*
_output_shapes
: *
T0
?
6My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/add/xConst"/device:GPU:1*
dtype0*
valueB
 *    *
_output_shapes
: 
?
4My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/addAddV26My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/add/x4My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul"/device:GPU:1*
T0*
_output_shapes
: 
?
+CCN_1Conv_x0/convB20/bias/Initializer/zerosConst*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
valueB?*    *
_output_shapes	
:?*
dtype0
?
CCN_1Conv_x0/convB20/bias
VariableV2"/device:GPU:1*
shape:?*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
dtype0
?
 CCN_1Conv_x0/convB20/bias/AssignAssignCCN_1Conv_x0/convB20/bias+CCN_1Conv_x0/convB20/bias/Initializer/zeros"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
T0*
_output_shapes	
:?
?
CCN_1Conv_x0/convB20/bias/readIdentityCCN_1Conv_x0/convB20/bias"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
+My_GPU_1/CCN_1Conv_x0/convB20/dilation_rateConst"/device:GPU:1*
dtype0*
valueB:*
_output_shapes
:
?
3My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims/dimConst"/device:GPU:1*
_output_shapes
: *
value	B :*
dtype0
?
/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims
ExpandDims%My_GPU_1/CCN_1Conv_x0/poolB10/Squeeze3My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims/dim"/device:GPU:1*
T0*0
_output_shapes
:??????????
?
5My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_1/dimConst"/device:GPU:1*
dtype0*
value	B : *
_output_shapes
: 
?
1My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_1
ExpandDims CCN_1Conv_x0/convB20/kernel/read5My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_1/dim"/device:GPU:1*
T0*(
_output_shapes
:??
?
$My_GPU_1/CCN_1Conv_x0/convB20/conv1dConv2D/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_1"/device:GPU:1*
paddingSAME*
T0*0
_output_shapes
:??????????*
strides

?
,My_GPU_1/CCN_1Conv_x0/convB20/conv1d/SqueezeSqueeze$My_GPU_1/CCN_1Conv_x0/convB20/conv1d"/device:GPU:1*
squeeze_dims
*,
_output_shapes
:??????????*
T0
?
%My_GPU_1/CCN_1Conv_x0/convB20/BiasAddBiasAdd,My_GPU_1/CCN_1Conv_x0/convB20/conv1d/SqueezeCCN_1Conv_x0/convB20/bias/read"/device:GPU:1*,
_output_shapes
:??????????*
T0
?
"My_GPU_1/CCN_1Conv_x0/convB20/ReluRelu%My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd"/device:GPU:1*,
_output_shapes
:??????????*
T0
?
<CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/shapeConst*!
valueB"         *
_output_shapes
:*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
dtype0
?
:CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
valueB
 *׳]?*
_output_shapes
: *
dtype0
?
:CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *׳]=*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
DCCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/RandomUniformRandomUniform<CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/shape*
T0*$
_output_shapes
:??*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
:CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/subSub:CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/max:CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/min*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0*
_output_shapes
: 
?
:CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/mulMulDCCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/RandomUniform:CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/sub*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
6CCN_1Conv_x0/convA11/kernel/Initializer/random_uniformAdd:CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/mul:CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform/min*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
CCN_1Conv_x0/convA11/kernel
VariableV2"/device:GPU:1*$
_output_shapes
:??*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
shape:??
?
"CCN_1Conv_x0/convA11/kernel/AssignAssignCCN_1Conv_x0/convA11/kernel6CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??
?
 CCN_1Conv_x0/convA11/kernel/readIdentityCCN_1Conv_x0/convA11/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??*
T0
?
7My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/SquareSquare CCN_1Conv_x0/convA11/kernel/read"/device:GPU:1*
T0*$
_output_shapes
:??
?
6My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/ConstConst"/device:GPU:1*
_output_shapes
:*
dtype0*!
valueB"          
?
4My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/SumSum7My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Square6My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Const"/device:GPU:1*
_output_shapes
: *
T0
?
6My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul/xConst"/device:GPU:1*
dtype0*
valueB
 *???3*
_output_shapes
: 
?
4My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mulMul6My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul/x4My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Sum"/device:GPU:1*
T0*
_output_shapes
: 
?
6My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/add/xConst"/device:GPU:1*
dtype0*
valueB
 *    *
_output_shapes
: 
?
4My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/addAddV26My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/add/x4My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul"/device:GPU:1*
_output_shapes
: *
T0
?
+CCN_1Conv_x0/convA11/bias/Initializer/zerosConst*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
dtype0*
valueB?*    *
_output_shapes	
:?
?
CCN_1Conv_x0/convA11/bias
VariableV2"/device:GPU:1*
_output_shapes	
:?*
shape:?*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias
?
 CCN_1Conv_x0/convA11/bias/AssignAssignCCN_1Conv_x0/convA11/bias+CCN_1Conv_x0/convA11/bias/Initializer/zeros"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?*
T0
?
CCN_1Conv_x0/convA11/bias/readIdentityCCN_1Conv_x0/convA11/bias"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?
?
+My_GPU_1/CCN_1Conv_x0/convA11/dilation_rateConst"/device:GPU:1*
dtype0*
_output_shapes
:*
valueB:
?
3My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims/dimConst"/device:GPU:1*
dtype0*
value	B :*
_output_shapes
: 
?
/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims
ExpandDims"My_GPU_1/CCN_1Conv_x0/convB20/Relu3My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims/dim"/device:GPU:1*0
_output_shapes
:??????????*
T0
?
5My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_1/dimConst"/device:GPU:1*
value	B : *
_output_shapes
: *
dtype0
?
1My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_1
ExpandDims CCN_1Conv_x0/convA11/kernel/read5My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_1/dim"/device:GPU:1*(
_output_shapes
:??*
T0
?
$My_GPU_1/CCN_1Conv_x0/convA11/conv1dConv2D/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_1"/device:GPU:1*
paddingSAME*
T0*
strides
*0
_output_shapes
:??????????
?
,My_GPU_1/CCN_1Conv_x0/convA11/conv1d/SqueezeSqueeze$My_GPU_1/CCN_1Conv_x0/convA11/conv1d"/device:GPU:1*
squeeze_dims
*
T0*,
_output_shapes
:??????????
?
%My_GPU_1/CCN_1Conv_x0/convA11/BiasAddBiasAdd,My_GPU_1/CCN_1Conv_x0/convA11/conv1d/SqueezeCCN_1Conv_x0/convA11/bias/read"/device:GPU:1*,
_output_shapes
:??????????*
T0
?
"My_GPU_1/CCN_1Conv_x0/convA11/ReluRelu%My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
<CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
dtype0*
_output_shapes
:*!
valueB"         
?
:CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
dtype0*
_output_shapes
: *
valueB
 *?5?
?
:CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/maxConst*
valueB
 *?5=*
_output_shapes
: *
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
DCCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/RandomUniformRandomUniform<CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/shape*
dtype0*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
:CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/subSub:CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/max:CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/min*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0*
_output_shapes
: 
?
:CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/mulMulDCCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/RandomUniform:CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*$
_output_shapes
:??
?
6CCN_1Conv_x0/convB11/kernel/Initializer/random_uniformAdd:CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/mul:CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform/min*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0
?
CCN_1Conv_x0/convB11/kernel
VariableV2"/device:GPU:1*
shape:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
dtype0*$
_output_shapes
:??
?
"CCN_1Conv_x0/convB11/kernel/AssignAssignCCN_1Conv_x0/convB11/kernel6CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0*$
_output_shapes
:??
?
 CCN_1Conv_x0/convB11/kernel/readIdentityCCN_1Conv_x0/convB11/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*$
_output_shapes
:??*
T0
?
7My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/SquareSquare CCN_1Conv_x0/convB11/kernel/read"/device:GPU:1*$
_output_shapes
:??*
T0
?
6My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/ConstConst"/device:GPU:1*!
valueB"          *
_output_shapes
:*
dtype0
?
4My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/SumSum7My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Square6My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Const"/device:GPU:1*
T0*
_output_shapes
: 
?
6My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul/xConst"/device:GPU:1*
dtype0*
valueB
 *???3*
_output_shapes
: 
?
4My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mulMul6My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul/x4My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Sum"/device:GPU:1*
_output_shapes
: *
T0
?
6My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/add/xConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *    
?
4My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/addAddV26My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/add/x4My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul"/device:GPU:1*
_output_shapes
: *
T0
?
+CCN_1Conv_x0/convB11/bias/Initializer/zerosConst*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
valueB?*    *
dtype0
?
CCN_1Conv_x0/convB11/bias
VariableV2"/device:GPU:1*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
shape:?*
_output_shapes	
:?
?
 CCN_1Conv_x0/convB11/bias/AssignAssignCCN_1Conv_x0/convB11/bias+CCN_1Conv_x0/convB11/bias/Initializer/zeros"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?*
T0
?
CCN_1Conv_x0/convB11/bias/readIdentityCCN_1Conv_x0/convB11/bias"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
+My_GPU_1/CCN_1Conv_x0/convB11/dilation_rateConst"/device:GPU:1*
_output_shapes
:*
valueB:*
dtype0
?
3My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims/dimConst"/device:GPU:1*
dtype0*
_output_shapes
: *
value	B :
?
/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims
ExpandDims"My_GPU_1/CCN_1Conv_x0/convA11/Relu3My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims/dim"/device:GPU:1*
T0*0
_output_shapes
:??????????
?
5My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_1/dimConst"/device:GPU:1*
value	B : *
_output_shapes
: *
dtype0
?
1My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_1
ExpandDims CCN_1Conv_x0/convB11/kernel/read5My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_1/dim"/device:GPU:1*(
_output_shapes
:??*
T0
?
$My_GPU_1/CCN_1Conv_x0/convB11/conv1dConv2D/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_1"/device:GPU:1*
strides
*
T0*0
_output_shapes
:??????????*
paddingSAME
?
,My_GPU_1/CCN_1Conv_x0/convB11/conv1d/SqueezeSqueeze$My_GPU_1/CCN_1Conv_x0/convB11/conv1d"/device:GPU:1*
squeeze_dims
*,
_output_shapes
:??????????*
T0
?
%My_GPU_1/CCN_1Conv_x0/convB11/BiasAddBiasAdd,My_GPU_1/CCN_1Conv_x0/convB11/conv1d/SqueezeCCN_1Conv_x0/convB11/bias/read"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
"My_GPU_1/CCN_1Conv_x0/convB11/ReluRelu%My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
+My_GPU_1/CCN_1Conv_x0/poolB11/dilation_rateConst"/device:GPU:1*
dtype0*
valueB:*
_output_shapes
:
}
,My_GPU_1/CCN_1Conv_x0/poolB11/ExpandDims/dimConst"/device:GPU:1*
value	B :*
_output_shapes
: *
dtype0
?
(My_GPU_1/CCN_1Conv_x0/poolB11/ExpandDims
ExpandDims"My_GPU_1/CCN_1Conv_x0/convB11/Relu,My_GPU_1/CCN_1Conv_x0/poolB11/ExpandDims/dim"/device:GPU:1*
T0*0
_output_shapes
:??????????
?
My_GPU_1/CCN_1Conv_x0/poolB11MaxPool(My_GPU_1/CCN_1Conv_x0/poolB11/ExpandDims"/device:GPU:1*0
_output_shapes
:??????????*
ksize
*
strides
*
paddingSAME
?
%My_GPU_1/CCN_1Conv_x0/poolB11/SqueezeSqueezeMy_GPU_1/CCN_1Conv_x0/poolB11"/device:GPU:1*
T0*,
_output_shapes
:??????????*
squeeze_dims

?
<CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
_output_shapes
:*
dtype0*!
valueB"         
?
:CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/minConst*
valueB
 *?5?*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
dtype0
?
:CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/maxConst*
valueB
 *?5=*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
dtype0
?
DCCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/RandomUniformRandomUniform<CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??*
dtype0
?
:CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/subSub:CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/max:CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/min*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
_output_shapes
: *
T0
?
:CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/mulMulDCCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/RandomUniform:CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??
?
6CCN_1Conv_x0/convB21/kernel/Initializer/random_uniformAdd:CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/mul:CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform/min*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??*
T0
?
CCN_1Conv_x0/convB21/kernel
VariableV2"/device:GPU:1*
shape:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
dtype0*$
_output_shapes
:??
?
"CCN_1Conv_x0/convB21/kernel/AssignAssignCCN_1Conv_x0/convB21/kernel6CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
T0*$
_output_shapes
:??
?
 CCN_1Conv_x0/convB21/kernel/readIdentityCCN_1Conv_x0/convB21/kernel"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
7My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/SquareSquare CCN_1Conv_x0/convB21/kernel/read"/device:GPU:1*
T0*$
_output_shapes
:??
?
6My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/ConstConst"/device:GPU:1*!
valueB"          *
_output_shapes
:*
dtype0
?
4My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/SumSum7My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Square6My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Const"/device:GPU:1*
_output_shapes
: *
T0
?
6My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul/xConst"/device:GPU:1*
dtype0*
valueB
 *???3*
_output_shapes
: 
?
4My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mulMul6My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul/x4My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Sum"/device:GPU:1*
_output_shapes
: *
T0
?
6My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/add/xConst"/device:GPU:1*
valueB
 *    *
_output_shapes
: *
dtype0
?
4My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/addAddV26My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/add/x4My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul"/device:GPU:1*
T0*
_output_shapes
: 
?
+CCN_1Conv_x0/convB21/bias/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
CCN_1Conv_x0/convB21/bias
VariableV2"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
dtype0*
shape:?*
_output_shapes	
:?
?
 CCN_1Conv_x0/convB21/bias/AssignAssignCCN_1Conv_x0/convB21/bias+CCN_1Conv_x0/convB21/bias/Initializer/zeros"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes	
:?*
T0
?
CCN_1Conv_x0/convB21/bias/readIdentityCCN_1Conv_x0/convB21/bias"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0
?
+My_GPU_1/CCN_1Conv_x0/convB21/dilation_rateConst"/device:GPU:1*
valueB:*
_output_shapes
:*
dtype0
?
3My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims/dimConst"/device:GPU:1*
value	B :*
_output_shapes
: *
dtype0
?
/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims
ExpandDims%My_GPU_1/CCN_1Conv_x0/poolB11/Squeeze3My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims/dim"/device:GPU:1*0
_output_shapes
:??????????*
T0
?
5My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_1/dimConst"/device:GPU:1*
value	B : *
_output_shapes
: *
dtype0
?
1My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_1
ExpandDims CCN_1Conv_x0/convB21/kernel/read5My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_1/dim"/device:GPU:1*(
_output_shapes
:??*
T0
?
$My_GPU_1/CCN_1Conv_x0/convB21/conv1dConv2D/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_1"/device:GPU:1*
paddingSAME*0
_output_shapes
:??????????*
strides
*
T0
?
,My_GPU_1/CCN_1Conv_x0/convB21/conv1d/SqueezeSqueeze$My_GPU_1/CCN_1Conv_x0/convB21/conv1d"/device:GPU:1*
squeeze_dims
*
T0*,
_output_shapes
:??????????
?
%My_GPU_1/CCN_1Conv_x0/convB21/BiasAddBiasAdd,My_GPU_1/CCN_1Conv_x0/convB21/conv1d/SqueezeCCN_1Conv_x0/convB21/bias/read"/device:GPU:1*,
_output_shapes
:??????????*
T0
?
"My_GPU_1/CCN_1Conv_x0/convB21/ReluRelu%My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
#My_GPU_1/CCN_1Conv_x0/Reshape/shapeConst"/device:GPU:1*
valueB"????   *
dtype0*
_output_shapes
:
?
My_GPU_1/CCN_1Conv_x0/ReshapeReshape"My_GPU_1/CCN_1Conv_x0/convB21/Relu#My_GPU_1/CCN_1Conv_x0/Reshape/shape"/device:GPU:1*(
_output_shapes
:??????????*
T0
k
My_GPU_1/concat/concat_dimConst"/device:GPU:1*
value	B :*
dtype0*
_output_shapes
: 
?
My_GPU_1/concat/concatIdentityMy_GPU_1/CCN_1Conv_x0/Reshape"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
1Conv_out__/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB:?*
_output_shapes
:*"
_class
loc:@Conv_out__/beta
?
'Conv_out__/beta/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: *"
_class
loc:@Conv_out__/beta
?
!Conv_out__/beta/Initializer/zerosFill1Conv_out__/beta/Initializer/zeros/shape_as_tensor'Conv_out__/beta/Initializer/zeros/Const*
T0*"
_class
loc:@Conv_out__/beta*
_output_shapes	
:?
?
Conv_out__/beta
VariableV2"/device:GPU:1*
dtype0*"
_class
loc:@Conv_out__/beta*
shape:?*
_output_shapes	
:?
?
Conv_out__/beta/AssignAssignConv_out__/beta!Conv_out__/beta/Initializer/zeros"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
T0*
_output_shapes	
:?
?
Conv_out__/beta/readIdentityConv_out__/beta"/device:GPU:1*
_output_shapes	
:?*
T0*"
_class
loc:@Conv_out__/beta
?
1Conv_out__/gamma/Initializer/ones/shape_as_tensorConst*
valueB:?*
_output_shapes
:*
dtype0*#
_class
loc:@Conv_out__/gamma
?
'Conv_out__/gamma/Initializer/ones/ConstConst*
valueB
 *  ??*
dtype0*#
_class
loc:@Conv_out__/gamma*
_output_shapes
: 
?
!Conv_out__/gamma/Initializer/onesFill1Conv_out__/gamma/Initializer/ones/shape_as_tensor'Conv_out__/gamma/Initializer/ones/Const*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma*
T0
?
Conv_out__/gamma
VariableV2"/device:GPU:1*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma*
shape:?*
dtype0
?
Conv_out__/gamma/AssignAssignConv_out__/gamma!Conv_out__/gamma/Initializer/ones"/device:GPU:1*
_output_shapes	
:?*
T0*#
_class
loc:@Conv_out__/gamma
?
Conv_out__/gamma/readIdentityConv_out__/gamma"/device:GPU:1*
T0*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma
?
=My_GPU_1/Conv_out__/Conv_out__/moments/mean/reduction_indicesConst"/device:GPU:1*
_output_shapes
:*
valueB:*
dtype0
?
+My_GPU_1/Conv_out__/Conv_out__/moments/meanMeanMy_GPU_1/concat/concat=My_GPU_1/Conv_out__/Conv_out__/moments/mean/reduction_indices"/device:GPU:1*
T0*'
_output_shapes
:?????????*
	keep_dims(
?
3My_GPU_1/Conv_out__/Conv_out__/moments/StopGradientStopGradient+My_GPU_1/Conv_out__/Conv_out__/moments/mean"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
8My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifferenceSquaredDifferenceMy_GPU_1/concat/concat3My_GPU_1/Conv_out__/Conv_out__/moments/StopGradient"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
AMy_GPU_1/Conv_out__/Conv_out__/moments/variance/reduction_indicesConst"/device:GPU:1*
_output_shapes
:*
dtype0*
valueB:
?
/My_GPU_1/Conv_out__/Conv_out__/moments/varianceMean8My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifferenceAMy_GPU_1/Conv_out__/Conv_out__/moments/variance/reduction_indices"/device:GPU:1*
T0*
	keep_dims(*'
_output_shapes
:?????????
?
.My_GPU_1/Conv_out__/Conv_out__/batchnorm/add/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *̼?+
?
,My_GPU_1/Conv_out__/Conv_out__/batchnorm/addAddV2/My_GPU_1/Conv_out__/Conv_out__/moments/variance.My_GPU_1/Conv_out__/Conv_out__/batchnorm/add/y"/device:GPU:1*
T0*'
_output_shapes
:?????????
?
.My_GPU_1/Conv_out__/Conv_out__/batchnorm/RsqrtRsqrt,My_GPU_1/Conv_out__/Conv_out__/batchnorm/add"/device:GPU:1*
T0*'
_output_shapes
:?????????
?
,My_GPU_1/Conv_out__/Conv_out__/batchnorm/mulMul.My_GPU_1/Conv_out__/Conv_out__/batchnorm/RsqrtConv_out__/gamma/read"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
.My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1MulMy_GPU_1/concat/concat,My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
.My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2Mul+My_GPU_1/Conv_out__/Conv_out__/moments/mean,My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
,My_GPU_1/Conv_out__/Conv_out__/batchnorm/subSubConv_out__/beta/read.My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
.My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1AddV2.My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1,My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
#My_GPU_1/Conv_out__/Conv_out__/ReluRelu.My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1"/device:GPU:1*
T0*(
_output_shapes
:??????????
m
My_GPU_1/Conv_out__/sub/xConst"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
q
My_GPU_1/Conv_out__/subSubMy_GPU_1/Conv_out__/sub/xkeep"/device:GPU:1*
_output_shapes
:*
T0
?
!My_GPU_1/Conv_out__/dropout/ShapeShape#My_GPU_1/Conv_out__/Conv_out__/Relu"/device:GPU:1*
T0*
_output_shapes
:
?
.My_GPU_1/Conv_out__/dropout/random_uniform/minConst"/device:GPU:1*
dtype0*
valueB
 *    *
_output_shapes
: 
?
.My_GPU_1/Conv_out__/dropout/random_uniform/maxConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
8My_GPU_1/Conv_out__/dropout/random_uniform/RandomUniformRandomUniform!My_GPU_1/Conv_out__/dropout/Shape"/device:GPU:1*
dtype0*
T0*(
_output_shapes
:??????????
?
.My_GPU_1/Conv_out__/dropout/random_uniform/subSub.My_GPU_1/Conv_out__/dropout/random_uniform/max.My_GPU_1/Conv_out__/dropout/random_uniform/min"/device:GPU:1*
_output_shapes
: *
T0
?
.My_GPU_1/Conv_out__/dropout/random_uniform/mulMul8My_GPU_1/Conv_out__/dropout/random_uniform/RandomUniform.My_GPU_1/Conv_out__/dropout/random_uniform/sub"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
*My_GPU_1/Conv_out__/dropout/random_uniformAdd.My_GPU_1/Conv_out__/dropout/random_uniform/mul.My_GPU_1/Conv_out__/dropout/random_uniform/min"/device:GPU:1*
T0*(
_output_shapes
:??????????
u
!My_GPU_1/Conv_out__/dropout/sub/xConst"/device:GPU:1*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
My_GPU_1/Conv_out__/dropout/subSub!My_GPU_1/Conv_out__/dropout/sub/xMy_GPU_1/Conv_out__/sub"/device:GPU:1*
_output_shapes
:*
T0
y
%My_GPU_1/Conv_out__/dropout/truediv/xConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
#My_GPU_1/Conv_out__/dropout/truedivRealDiv%My_GPU_1/Conv_out__/dropout/truediv/xMy_GPU_1/Conv_out__/dropout/sub"/device:GPU:1*
T0*
_output_shapes
:
?
(My_GPU_1/Conv_out__/dropout/GreaterEqualGreaterEqual*My_GPU_1/Conv_out__/dropout/random_uniformMy_GPU_1/Conv_out__/sub"/device:GPU:1*
_output_shapes
:*
T0
?
My_GPU_1/Conv_out__/dropout/mulMul#My_GPU_1/Conv_out__/Conv_out__/Relu#My_GPU_1/Conv_out__/dropout/truediv"/device:GPU:1*
T0*
_output_shapes
:
?
 My_GPU_1/Conv_out__/dropout/CastCast(My_GPU_1/Conv_out__/dropout/GreaterEqual"/device:GPU:1*

SrcT0
*
_output_shapes
:*

DstT0
?
!My_GPU_1/Conv_out__/dropout/mul_1MulMy_GPU_1/Conv_out__/dropout/mul My_GPU_1/Conv_out__/dropout/Cast"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
CReconstruction_Output/dense/kernel/Initializer/random_uniform/shapeConst*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:*
valueB"      *
dtype0
?
AReconstruction_Output/dense/kernel/Initializer/random_uniform/minConst*
dtype0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
: *
valueB
 *????
?
AReconstruction_Output/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *???=*
dtype0*
_output_shapes
: *5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
KReconstruction_Output/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniformCReconstruction_Output/dense/kernel/Initializer/random_uniform/shape*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
dtype0*
_output_shapes
:	?
?
AReconstruction_Output/dense/kernel/Initializer/random_uniform/subSubAReconstruction_Output/dense/kernel/Initializer/random_uniform/maxAReconstruction_Output/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
AReconstruction_Output/dense/kernel/Initializer/random_uniform/mulMulKReconstruction_Output/dense/kernel/Initializer/random_uniform/RandomUniformAReconstruction_Output/dense/kernel/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?
?
=Reconstruction_Output/dense/kernel/Initializer/random_uniformAddAReconstruction_Output/dense/kernel/Initializer/random_uniform/mulAReconstruction_Output/dense/kernel/Initializer/random_uniform/min*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
T0*
_output_shapes
:	?
?
"Reconstruction_Output/dense/kernel
VariableV2"/device:GPU:1*
_output_shapes
:	?*
dtype0*
shape:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
)Reconstruction_Output/dense/kernel/AssignAssign"Reconstruction_Output/dense/kernel=Reconstruction_Output/dense/kernel/Initializer/random_uniform"/device:GPU:1*
T0*
_output_shapes
:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
'Reconstruction_Output/dense/kernel/readIdentity"Reconstruction_Output/dense/kernel"/device:GPU:1*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?
?
>My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/SquareSquare'Reconstruction_Output/dense/kernel/read"/device:GPU:1*
_output_shapes
:	?*
T0
?
=My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/ConstConst"/device:GPU:1*
dtype0*
valueB"       *
_output_shapes
:
?
;My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/SumSum>My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Square=My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Const"/device:GPU:1*
_output_shapes
: *
T0
?
=My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul/xConst"/device:GPU:1*
valueB
 *???3*
_output_shapes
: *
dtype0
?
;My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mulMul=My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul/x;My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Sum"/device:GPU:1*
T0*
_output_shapes
: 
?
=My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/add/xConst"/device:GPU:1*
valueB
 *    *
_output_shapes
: *
dtype0
?
;My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/addAddV2=My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/add/x;My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul"/device:GPU:1*
T0*
_output_shapes
: 
?
2Reconstruction_Output/dense/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
 Reconstruction_Output/dense/bias
VariableV2"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
shape:*
dtype0*
_output_shapes
:
?
'Reconstruction_Output/dense/bias/AssignAssign Reconstruction_Output/dense/bias2Reconstruction_Output/dense/bias/Initializer/zeros"/device:GPU:1*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
:
?
%Reconstruction_Output/dense/bias/readIdentity Reconstruction_Output/dense/bias"/device:GPU:1*
_output_shapes
:*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
T0
?
+My_GPU_1/Reconstruction_Output/dense/MatMulMatMul#My_GPU_1/Conv_out__/Conv_out__/Relu'Reconstruction_Output/dense/kernel/read"/device:GPU:1*
T0*'
_output_shapes
:?????????
?
,My_GPU_1/Reconstruction_Output/dense/BiasAddBiasAdd+My_GPU_1/Reconstruction_Output/dense/MatMul%Reconstruction_Output/dense/bias/read"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"       *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
?
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *?_??*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0
?
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
dtype0*
valueB
 *?_?=*
_output_shapes
: 
?
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
_output_shapes
:	? *
T0*
_class
loc:@dense/kernel*
dtype0
?
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes
: 
?
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	? *
T0*
_class
loc:@dense/kernel
?
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes
:	? 
?
dense/kernel
VariableV2"/device:GPU:1*
shape:	? *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	? 
?
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform"/device:GPU:1*
_class
loc:@dense/kernel*
T0*
_output_shapes
:	? 
?
dense/kernel/readIdentitydense/kernel"/device:GPU:1*
_class
loc:@dense/kernel*
_output_shapes
:	? *
T0
~
(My_GPU_1/dense/kernel/Regularizer/SquareSquaredense/kernel/read"/device:GPU:1*
T0*
_output_shapes
:	? 
?
'My_GPU_1/dense/kernel/Regularizer/ConstConst"/device:GPU:1*
_output_shapes
:*
valueB"       *
dtype0
?
%My_GPU_1/dense/kernel/Regularizer/SumSum(My_GPU_1/dense/kernel/Regularizer/Square'My_GPU_1/dense/kernel/Regularizer/Const"/device:GPU:1*
_output_shapes
: *
T0
{
'My_GPU_1/dense/kernel/Regularizer/mul/xConst"/device:GPU:1*
dtype0*
valueB
 *???3*
_output_shapes
: 
?
%My_GPU_1/dense/kernel/Regularizer/mulMul'My_GPU_1/dense/kernel/Regularizer/mul/x%My_GPU_1/dense/kernel/Regularizer/Sum"/device:GPU:1*
_output_shapes
: *
T0
{
'My_GPU_1/dense/kernel/Regularizer/add/xConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *    
?
%My_GPU_1/dense/kernel/Regularizer/addAddV2'My_GPU_1/dense/kernel/Regularizer/add/x%My_GPU_1/dense/kernel/Regularizer/mul"/device:GPU:1*
_output_shapes
: *
T0
?
dense/bias/Initializer/zerosConst*
_output_shapes
: *
dtype0*
_class
loc:@dense/bias*
valueB *    
?

dense/bias
VariableV2"/device:GPU:1*
dtype0*
shape: *
_class
loc:@dense/bias*
_output_shapes
: 
?
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros"/device:GPU:1*
_class
loc:@dense/bias*
_output_shapes
: *
T0
z
dense/bias/readIdentity
dense/bias"/device:GPU:1*
_class
loc:@dense/bias*
_output_shapes
: *
T0
?
My_GPU_1/dense/MatMulMatMul!My_GPU_1/Conv_out__/dropout/mul_1dense/kernel/read"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
My_GPU_1/dense/BiasAddBiasAddMy_GPU_1/dense/MatMuldense/bias/read"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
?FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"        *
_output_shapes
:*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
=FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/minConst*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
?
=FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *qĜ>*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
dtype0
?
GFCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform?FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/shape*
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes

:  *
dtype0
?
=FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/subSub=FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/max=FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes
: 
?
=FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/mulMulGFCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/RandomUniform=FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/sub*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
T0*
_output_shapes

:  
?
9FCU_muiltDense_x0/dense/kernel/Initializer/random_uniformAdd=FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/mul=FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform/min*
_output_shapes

:  *
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
FCU_muiltDense_x0/dense/kernel
VariableV2"/device:GPU:1*
shape
:  *
_output_shapes

:  *
dtype0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
%FCU_muiltDense_x0/dense/kernel/AssignAssignFCU_muiltDense_x0/dense/kernel9FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform"/device:GPU:1*
_output_shapes

:  *
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
#FCU_muiltDense_x0/dense/kernel/readIdentityFCU_muiltDense_x0/dense/kernel"/device:GPU:1*
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes

:  
?
:My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/SquareSquare#FCU_muiltDense_x0/dense/kernel/read"/device:GPU:1*
_output_shapes

:  *
T0
?
9My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/ConstConst"/device:GPU:1*
valueB"       *
_output_shapes
:*
dtype0
?
7My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/SumSum:My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Square9My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Const"/device:GPU:1*
_output_shapes
: *
T0
?
9My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul/xConst"/device:GPU:1*
dtype0*
valueB
 *???3*
_output_shapes
: 
?
7My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mulMul9My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul/x7My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Sum"/device:GPU:1*
T0*
_output_shapes
: 
?
9My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/add/xConst"/device:GPU:1*
valueB
 *    *
dtype0*
_output_shapes
: 
?
7My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/addAddV29My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/add/x7My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul"/device:GPU:1*
T0*
_output_shapes
: 
?
.FCU_muiltDense_x0/dense/bias/Initializer/zerosConst*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
valueB *    *
dtype0*
_output_shapes
: 
?
FCU_muiltDense_x0/dense/bias
VariableV2"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
shape: *
_output_shapes
: *
dtype0
?
#FCU_muiltDense_x0/dense/bias/AssignAssignFCU_muiltDense_x0/dense/bias.FCU_muiltDense_x0/dense/bias/Initializer/zeros"/device:GPU:1*
T0*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
!FCU_muiltDense_x0/dense/bias/readIdentityFCU_muiltDense_x0/dense/bias"/device:GPU:1*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0
?
'My_GPU_1/FCU_muiltDense_x0/dense/MatMulMatMulMy_GPU_1/dense/BiasAdd#FCU_muiltDense_x0/dense/kernel/read"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
(My_GPU_1/FCU_muiltDense_x0/dense/BiasAddBiasAdd'My_GPU_1/FCU_muiltDense_x0/dense/MatMul!FCU_muiltDense_x0/dense/bias/read"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
(FCU_muiltDense_x0/beta/Initializer/zerosConst*
valueB *    *
_output_shapes
: *
dtype0*)
_class
loc:@FCU_muiltDense_x0/beta
?
FCU_muiltDense_x0/beta
VariableV2"/device:GPU:1*
shape: *
_output_shapes
: *
dtype0*)
_class
loc:@FCU_muiltDense_x0/beta
?
FCU_muiltDense_x0/beta/AssignAssignFCU_muiltDense_x0/beta(FCU_muiltDense_x0/beta/Initializer/zeros"/device:GPU:1*
_output_shapes
: *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
FCU_muiltDense_x0/beta/readIdentityFCU_muiltDense_x0/beta"/device:GPU:1*
_output_shapes
: *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
(FCU_muiltDense_x0/gamma/Initializer/onesConst*
dtype0**
_class 
loc:@FCU_muiltDense_x0/gamma*
valueB *  ??*
_output_shapes
: 
?
FCU_muiltDense_x0/gamma
VariableV2"/device:GPU:1*
shape: *
_output_shapes
: *
dtype0**
_class 
loc:@FCU_muiltDense_x0/gamma
?
FCU_muiltDense_x0/gamma/AssignAssignFCU_muiltDense_x0/gamma(FCU_muiltDense_x0/gamma/Initializer/ones"/device:GPU:1*
_output_shapes
: *
T0**
_class 
loc:@FCU_muiltDense_x0/gamma
?
FCU_muiltDense_x0/gamma/readIdentityFCU_muiltDense_x0/gamma"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: *
T0
?
KMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean/reduction_indicesConst"/device:GPU:1*
dtype0*
valueB:*
_output_shapes
:
?
9My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/meanMean(My_GPU_1/FCU_muiltDense_x0/dense/BiasAddKMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean/reduction_indices"/device:GPU:1*
T0*'
_output_shapes
:?????????*
	keep_dims(
?
AMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/StopGradientStopGradient9My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
FMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifferenceSquaredDifference(My_GPU_1/FCU_muiltDense_x0/dense/BiasAddAMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/StopGradient"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
OMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance/reduction_indicesConst"/device:GPU:1*
dtype0*
valueB:*
_output_shapes
:
?
=My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/varianceMeanFMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifferenceOMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance/reduction_indices"/device:GPU:1*
	keep_dims(*'
_output_shapes
:?????????*
T0
?
<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *̼?+*
dtype0
?
:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/addAddV2=My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add/y"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/RsqrtRsqrt:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add"/device:GPU:1*
T0*'
_output_shapes
:?????????
?
:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mulMul<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/RsqrtFCU_muiltDense_x0/gamma/read"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1Mul(My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2Mul9My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/subSubFCU_muiltDense_x0/beta/read<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1AddV2<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
1My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/ReluRelu<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1"/device:GPU:1*'
_output_shapes
:????????? *
T0
t
 My_GPU_1/FCU_muiltDense_x0/sub/xConst"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0

My_GPU_1/FCU_muiltDense_x0/subSub My_GPU_1/FCU_muiltDense_x0/sub/xkeep"/device:GPU:1*
T0*
_output_shapes
:
?
(My_GPU_1/FCU_muiltDense_x0/dropout/ShapeShape1My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/Relu"/device:GPU:1*
_output_shapes
:*
T0
?
5My_GPU_1/FCU_muiltDense_x0/dropout/random_uniform/minConst"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB
 *    
?
5My_GPU_1/FCU_muiltDense_x0/dropout/random_uniform/maxConst"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
?My_GPU_1/FCU_muiltDense_x0/dropout/random_uniform/RandomUniformRandomUniform(My_GPU_1/FCU_muiltDense_x0/dropout/Shape"/device:GPU:1*'
_output_shapes
:????????? *
dtype0*
T0
?
5My_GPU_1/FCU_muiltDense_x0/dropout/random_uniform/subSub5My_GPU_1/FCU_muiltDense_x0/dropout/random_uniform/max5My_GPU_1/FCU_muiltDense_x0/dropout/random_uniform/min"/device:GPU:1*
_output_shapes
: *
T0
?
5My_GPU_1/FCU_muiltDense_x0/dropout/random_uniform/mulMul?My_GPU_1/FCU_muiltDense_x0/dropout/random_uniform/RandomUniform5My_GPU_1/FCU_muiltDense_x0/dropout/random_uniform/sub"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
1My_GPU_1/FCU_muiltDense_x0/dropout/random_uniformAdd5My_GPU_1/FCU_muiltDense_x0/dropout/random_uniform/mul5My_GPU_1/FCU_muiltDense_x0/dropout/random_uniform/min"/device:GPU:1*'
_output_shapes
:????????? *
T0
|
(My_GPU_1/FCU_muiltDense_x0/dropout/sub/xConst"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
&My_GPU_1/FCU_muiltDense_x0/dropout/subSub(My_GPU_1/FCU_muiltDense_x0/dropout/sub/xMy_GPU_1/FCU_muiltDense_x0/sub"/device:GPU:1*
_output_shapes
:*
T0
?
,My_GPU_1/FCU_muiltDense_x0/dropout/truediv/xConst"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
*My_GPU_1/FCU_muiltDense_x0/dropout/truedivRealDiv,My_GPU_1/FCU_muiltDense_x0/dropout/truediv/x&My_GPU_1/FCU_muiltDense_x0/dropout/sub"/device:GPU:1*
T0*
_output_shapes
:
?
/My_GPU_1/FCU_muiltDense_x0/dropout/GreaterEqualGreaterEqual1My_GPU_1/FCU_muiltDense_x0/dropout/random_uniformMy_GPU_1/FCU_muiltDense_x0/sub"/device:GPU:1*
T0*
_output_shapes
:
?
&My_GPU_1/FCU_muiltDense_x0/dropout/mulMul1My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/Relu*My_GPU_1/FCU_muiltDense_x0/dropout/truediv"/device:GPU:1*
_output_shapes
:*
T0
?
'My_GPU_1/FCU_muiltDense_x0/dropout/CastCast/My_GPU_1/FCU_muiltDense_x0/dropout/GreaterEqual"/device:GPU:1*
_output_shapes
:*

SrcT0
*

DstT0
?
(My_GPU_1/FCU_muiltDense_x0/dropout/mul_1Mul&My_GPU_1/FCU_muiltDense_x0/dropout/mul'My_GPU_1/FCU_muiltDense_x0/dropout/Cast"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
My_GPU_1/addAddV2(My_GPU_1/FCU_muiltDense_x0/dropout/mul_1My_GPU_1/dense/BiasAdd"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
5Output_/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"       *'
_class
loc:@Output_/dense/kernel*
_output_shapes
:*
dtype0
?
3Output_/dense/kernel/Initializer/random_uniform/minConst*'
_class
loc:@Output_/dense/kernel*
valueB
 *JQھ*
_output_shapes
: *
dtype0
?
3Output_/dense/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@Output_/dense/kernel*
valueB
 *JQ?>*
_output_shapes
: *
dtype0
?
=Output_/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5Output_/dense/kernel/Initializer/random_uniform/shape*
dtype0*'
_class
loc:@Output_/dense/kernel*
T0*
_output_shapes

: 
?
3Output_/dense/kernel/Initializer/random_uniform/subSub3Output_/dense/kernel/Initializer/random_uniform/max3Output_/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@Output_/dense/kernel
?
3Output_/dense/kernel/Initializer/random_uniform/mulMul=Output_/dense/kernel/Initializer/random_uniform/RandomUniform3Output_/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@Output_/dense/kernel*
_output_shapes

: 
?
/Output_/dense/kernel/Initializer/random_uniformAdd3Output_/dense/kernel/Initializer/random_uniform/mul3Output_/dense/kernel/Initializer/random_uniform/min*'
_class
loc:@Output_/dense/kernel*
T0*
_output_shapes

: 
?
Output_/dense/kernel
VariableV2"/device:GPU:1*
_output_shapes

: *
shape
: *
dtype0*'
_class
loc:@Output_/dense/kernel
?
Output_/dense/kernel/AssignAssignOutput_/dense/kernel/Output_/dense/kernel/Initializer/random_uniform"/device:GPU:1*
_output_shapes

: *'
_class
loc:@Output_/dense/kernel*
T0
?
Output_/dense/kernel/readIdentityOutput_/dense/kernel"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*
_output_shapes

: *
T0
?
0My_GPU_1/Output_/dense/kernel/Regularizer/SquareSquareOutput_/dense/kernel/read"/device:GPU:1*
_output_shapes

: *
T0
?
/My_GPU_1/Output_/dense/kernel/Regularizer/ConstConst"/device:GPU:1*
valueB"       *
dtype0*
_output_shapes
:
?
-My_GPU_1/Output_/dense/kernel/Regularizer/SumSum0My_GPU_1/Output_/dense/kernel/Regularizer/Square/My_GPU_1/Output_/dense/kernel/Regularizer/Const"/device:GPU:1*
T0*
_output_shapes
: 
?
/My_GPU_1/Output_/dense/kernel/Regularizer/mul/xConst"/device:GPU:1*
valueB
 *???3*
dtype0*
_output_shapes
: 
?
-My_GPU_1/Output_/dense/kernel/Regularizer/mulMul/My_GPU_1/Output_/dense/kernel/Regularizer/mul/x-My_GPU_1/Output_/dense/kernel/Regularizer/Sum"/device:GPU:1*
_output_shapes
: *
T0
?
/My_GPU_1/Output_/dense/kernel/Regularizer/add/xConst"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB
 *    
?
-My_GPU_1/Output_/dense/kernel/Regularizer/addAddV2/My_GPU_1/Output_/dense/kernel/Regularizer/add/x-My_GPU_1/Output_/dense/kernel/Regularizer/mul"/device:GPU:1*
_output_shapes
: *
T0
?
$Output_/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*%
_class
loc:@Output_/dense/bias*
valueB*    
?
Output_/dense/bias
VariableV2"/device:GPU:1*
shape:*%
_class
loc:@Output_/dense/bias*
dtype0*
_output_shapes
:
?
Output_/dense/bias/AssignAssignOutput_/dense/bias$Output_/dense/bias/Initializer/zeros"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
T0*
_output_shapes
:
?
Output_/dense/bias/readIdentityOutput_/dense/bias"/device:GPU:1*
_output_shapes
:*
T0*%
_class
loc:@Output_/dense/bias
?
My_GPU_1/Output_/dense/MatMulMatMulMy_GPU_1/addOutput_/dense/kernel/read"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
My_GPU_1/Output_/dense/BiasAddBiasAddMy_GPU_1/Output_/dense/MatMulOutput_/dense/bias/read"/device:GPU:1*'
_output_shapes
:?????????*
T0
|
My_GPU_1/strided_slice/stackConst"/device:GPU:1*
_output_shapes
:*
valueB"       *
dtype0
~
My_GPU_1/strided_slice/stack_1Const"/device:GPU:1*
dtype0*
valueB"       *
_output_shapes
:
~
My_GPU_1/strided_slice/stack_2Const"/device:GPU:1*
dtype0*
valueB"      *
_output_shapes
:
?
My_GPU_1/strided_sliceStridedSliceIteratorGetNext:1My_GPU_1/strided_slice/stackMy_GPU_1/strided_slice/stack_1My_GPU_1/strided_slice/stack_2"/device:GPU:1*
Index0*
T0*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:?????????
~
My_GPU_1/strided_slice_1/stackConst"/device:GPU:1*
valueB"        *
_output_shapes
:*
dtype0
?
 My_GPU_1/strided_slice_1/stack_1Const"/device:GPU:1*
_output_shapes
:*
dtype0*
valueB"       
?
 My_GPU_1/strided_slice_1/stack_2Const"/device:GPU:1*
valueB"      *
_output_shapes
:*
dtype0
?
My_GPU_1/strided_slice_1StridedSliceMy_GPU_1/Output_/dense/BiasAddMy_GPU_1/strided_slice_1/stack My_GPU_1/strided_slice_1/stack_1 My_GPU_1/strided_slice_1/stack_2"/device:GPU:1*
Index0*
end_mask*

begin_mask*
shrink_axis_mask*#
_output_shapes
:?????????*
T0
?
-My_GPU_1/mean_squared_error/SquaredDifferenceSquaredDifferenceMy_GPU_1/strided_slice_1My_GPU_1/strided_slice"/device:GPU:1*#
_output_shapes
:?????????*
T0
?
8My_GPU_1/mean_squared_error/assert_broadcastable/weightsConst"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
>My_GPU_1/mean_squared_error/assert_broadcastable/weights/shapeConst"/device:GPU:1*
_output_shapes
: *
valueB *
dtype0
?
=My_GPU_1/mean_squared_error/assert_broadcastable/weights/rankConst"/device:GPU:1*
dtype0*
value	B : *
_output_shapes
: 
?
=My_GPU_1/mean_squared_error/assert_broadcastable/values/shapeShape-My_GPU_1/mean_squared_error/SquaredDifference"/device:GPU:1*
T0*
_output_shapes
:
?
<My_GPU_1/mean_squared_error/assert_broadcastable/values/rankConst"/device:GPU:1*
value	B :*
_output_shapes
: *
dtype0
c
LMy_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_successNoOp"/device:GPU:1
?
"My_GPU_1/mean_squared_error/Cast/xConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
My_GPU_1/mean_squared_error/MulMul-My_GPU_1/mean_squared_error/SquaredDifference"My_GPU_1/mean_squared_error/Cast/x"/device:GPU:1*#
_output_shapes
:?????????*
T0
?
!My_GPU_1/mean_squared_error/ConstConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
_output_shapes
:*
dtype0*
valueB: 
?
My_GPU_1/mean_squared_error/SumSumMy_GPU_1/mean_squared_error/Mul!My_GPU_1/mean_squared_error/Const"/device:GPU:1*
T0*
_output_shapes
: 
?
/My_GPU_1/mean_squared_error/num_present/Equal/yConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
valueB
 *    *
dtype0*
_output_shapes
: 
?
-My_GPU_1/mean_squared_error/num_present/EqualEqual"My_GPU_1/mean_squared_error/Cast/x/My_GPU_1/mean_squared_error/num_present/Equal/y"/device:GPU:1*
_output_shapes
: *
T0
?
2My_GPU_1/mean_squared_error/num_present/zeros_likeConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *    
?
7My_GPU_1/mean_squared_error/num_present/ones_like/ShapeConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB 
?
7My_GPU_1/mean_squared_error/num_present/ones_like/ConstConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
1My_GPU_1/mean_squared_error/num_present/ones_likeFill7My_GPU_1/mean_squared_error/num_present/ones_like/Shape7My_GPU_1/mean_squared_error/num_present/ones_like/Const"/device:GPU:1*
T0*
_output_shapes
: 
?
.My_GPU_1/mean_squared_error/num_present/SelectSelect-My_GPU_1/mean_squared_error/num_present/Equal2My_GPU_1/mean_squared_error/num_present/zeros_like1My_GPU_1/mean_squared_error/num_present/ones_like"/device:GPU:1*
_output_shapes
: *
T0
?
\My_GPU_1/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
dtype0*
valueB *
_output_shapes
: 
?
[My_GPU_1/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
_output_shapes
: *
dtype0*
value	B : 
?
[My_GPU_1/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape-My_GPU_1/mean_squared_error/SquaredDifferenceM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
T0*
_output_shapes
:
?
ZMy_GPU_1/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
dtype0*
_output_shapes
: *
value	B :
?
jMy_GPU_1/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1
?
IMy_GPU_1/mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape-My_GPU_1/mean_squared_error/SquaredDifferenceM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_successk^My_GPU_1/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
T0*
_output_shapes
:
?
IMy_GPU_1/mean_squared_error/num_present/broadcast_weights/ones_like/ConstConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_successk^My_GPU_1/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
CMy_GPU_1/mean_squared_error/num_present/broadcast_weights/ones_likeFillIMy_GPU_1/mean_squared_error/num_present/broadcast_weights/ones_like/ShapeIMy_GPU_1/mean_squared_error/num_present/broadcast_weights/ones_like/Const"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
9My_GPU_1/mean_squared_error/num_present/broadcast_weightsMul.My_GPU_1/mean_squared_error/num_present/SelectCMy_GPU_1/mean_squared_error/num_present/broadcast_weights/ones_like"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
-My_GPU_1/mean_squared_error/num_present/ConstConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
dtype0*
valueB: *
_output_shapes
:
?
'My_GPU_1/mean_squared_error/num_presentSum9My_GPU_1/mean_squared_error/num_present/broadcast_weights-My_GPU_1/mean_squared_error/num_present/Const"/device:GPU:1*
_output_shapes
: *
T0
?
#My_GPU_1/mean_squared_error/Const_1ConstM^My_GPU_1/mean_squared_error/assert_broadcastable/static_scalar_check_success"/device:GPU:1*
valueB *
dtype0*
_output_shapes
: 
?
!My_GPU_1/mean_squared_error/Sum_1SumMy_GPU_1/mean_squared_error/Sum#My_GPU_1/mean_squared_error/Const_1"/device:GPU:1*
_output_shapes
: *
T0
?
!My_GPU_1/mean_squared_error/valueDivNoNan!My_GPU_1/mean_squared_error/Sum_1'My_GPU_1/mean_squared_error/num_present"/device:GPU:1*
T0*
_output_shapes
: 
b
My_GPU_1/mul/yConst"/device:GPU:1*
valueB
 *?k@*
_output_shapes
: *
dtype0
v
My_GPU_1/mulMul!My_GPU_1/mean_squared_error/valueMy_GPU_1/mul/y"/device:GPU:1*
_output_shapes
: *
T0
r
My_GPU_1/Reshape/shape/1Const"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB :
?????????
{
My_GPU_1/Reshape/shapePackSizeMy_GPU_1/Reshape/shape/1"/device:GPU:1*
T0*
N*
_output_shapes
:
?
My_GPU_1/ReshapeReshapeReshapeMy_GPU_1/Reshape/shape"/device:GPU:1*0
_output_shapes
:??????????????????*
T0
?
My_GPU_1/subSubMy_GPU_1/Reshape,My_GPU_1/Reconstruction_Output/dense/BiasAdd"/device:GPU:1*
T0*'
_output_shapes
:?????????
h
My_GPU_1/SquareSquareMy_GPU_1/sub"/device:GPU:1*'
_output_shapes
:?????????*
T0
n
My_GPU_1/ConstConst"/device:GPU:1*
dtype0*
valueB"       *
_output_shapes
:
f
My_GPU_1/MeanMeanMy_GPU_1/SquareMy_GPU_1/Const"/device:GPU:1*
_output_shapes
: *
T0
d
My_GPU_1/mul_1/yConst"/device:GPU:1*
valueB
 *-	P=*
dtype0*
_output_shapes
: 
f
My_GPU_1/mul_1MulMy_GPU_1/MeanMy_GPU_1/mul_1/y"/device:GPU:1*
_output_shapes
: *
T0
?
My_GPU_1/PyFuncPyFuncMy_GPU_1/strided_sliceMy_GPU_1/strided_slice_1"/device:GPU:1*
token
pyfunc_0*
_output_shapes
:*
Tin
2*
Tout
2
d
My_GPU_1/mul_2/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *q=
?*
dtype0
j
My_GPU_1/mul_2MulMy_GPU_1/PyFuncMy_GPU_1/mul_2/y"/device:GPU:1*
T0*
_output_shapes
:
?
My_GPU_1/AddNAddN4My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/add4My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/add4My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/add4My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/add4My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/add4My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/add;My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/add%My_GPU_1/dense/kernel/Regularizer/add7My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/add-My_GPU_1/Output_/dense/kernel/Regularizer/add"/device:GPU:1*
_output_shapes
: *
T0*
N

d
My_GPU_1/add_1AddV2My_GPU_1/mulMy_GPU_1/AddN"/device:GPU:1*
_output_shapes
: *
T0
g
My_GPU_1/add_2AddV2My_GPU_1/add_1My_GPU_1/mul_1"/device:GPU:1*
T0*
_output_shapes
: 
i
My_GPU_1/add_3AddV2My_GPU_1/add_2My_GPU_1/mul_2"/device:GPU:1*
T0*
_output_shapes
:
{
My_GPU_1/Total_Loss/tagsConst"/device:GPU:1*$
valueB BMy_GPU_1/Total_Loss*
_output_shapes
: *
dtype0
~
My_GPU_1/Total_LossScalarSummaryMy_GPU_1/Total_Loss/tagsMy_GPU_1/add_3"/device:GPU:1*
_output_shapes
: *
T0
?
My_GPU_1/Main_loss_value/tagsConst"/device:GPU:1*
_output_shapes
: *
dtype0*)
value B BMy_GPU_1/Main_loss_value
?
My_GPU_1/Main_loss_valueScalarSummaryMy_GPU_1/Main_loss_value/tagsMy_GPU_1/mul"/device:GPU:1*
T0*
_output_shapes
: 

My_GPU_1/Loss_l2_loss/tagsConst"/device:GPU:1*
dtype0*
_output_shapes
: *&
valueB BMy_GPU_1/Loss_l2_loss
?
My_GPU_1/Loss_l2_lossScalarSummaryMy_GPU_1/Loss_l2_loss/tagsMy_GPU_1/AddN"/device:GPU:1*
T0*
_output_shapes
: 
?
My_GPU_1/total_loss_1AddN!My_GPU_1/mean_squared_error/valueMy_GPU_1/add_3"/device:GPU:1*
_output_shapes
: *
T0*
N
j
My_GPU_1/gradients/ShapeConst"/device:GPU:1*
dtype0*
valueB *
_output_shapes
: 
p
My_GPU_1/gradients/grad_ys_0Const"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
My_GPU_1/gradients/FillFillMy_GPU_1/gradients/ShapeMy_GPU_1/gradients/grad_ys_0"/device:GPU:1*
T0*
_output_shapes
: 
o
>My_GPU_1/gradients/My_GPU_1/total_loss_1_grad/tuple/group_depsNoOp^My_GPU_1/gradients/Fill"/device:GPU:1
?
FMy_GPU_1/gradients/My_GPU_1/total_loss_1_grad/tuple/control_dependencyIdentityMy_GPU_1/gradients/Fill?^My_GPU_1/gradients/My_GPU_1/total_loss_1_grad/tuple/group_deps"/device:GPU:1**
_class 
loc:@My_GPU_1/gradients/Fill*
T0*
_output_shapes
: 
?
HMy_GPU_1/gradients/My_GPU_1/total_loss_1_grad/tuple/control_dependency_1IdentityMy_GPU_1/gradients/Fill?^My_GPU_1/gradients/My_GPU_1/total_loss_1_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0**
_class 
loc:@My_GPU_1/gradients/Fill
w
,My_GPU_1/gradients/My_GPU_1/add_3_grad/ShapeShapeMy_GPU_1/add_2"/device:GPU:1*
T0*
_output_shapes
: 
?
.My_GPU_1/gradients/My_GPU_1/add_3_grad/Shape_1ShapeMy_GPU_1/mul_2"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
<My_GPU_1/gradients/My_GPU_1/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs,My_GPU_1/gradients/My_GPU_1/add_3_grad/Shape.My_GPU_1/gradients/My_GPU_1/add_3_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
*My_GPU_1/gradients/My_GPU_1/add_3_grad/SumSumHMy_GPU_1/gradients/My_GPU_1/total_loss_1_grad/tuple/control_dependency_1<My_GPU_1/gradients/My_GPU_1/add_3_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
.My_GPU_1/gradients/My_GPU_1/add_3_grad/ReshapeReshape*My_GPU_1/gradients/My_GPU_1/add_3_grad/Sum,My_GPU_1/gradients/My_GPU_1/add_3_grad/Shape"/device:GPU:1*
T0*
_output_shapes
: 
?
,My_GPU_1/gradients/My_GPU_1/add_3_grad/Sum_1SumHMy_GPU_1/gradients/My_GPU_1/total_loss_1_grad/tuple/control_dependency_1>My_GPU_1/gradients/My_GPU_1/add_3_grad/BroadcastGradientArgs:1"/device:GPU:1*
T0*
_output_shapes
:
?
0My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape_1Reshape,My_GPU_1/gradients/My_GPU_1/add_3_grad/Sum_1.My_GPU_1/gradients/My_GPU_1/add_3_grad/Shape_1"/device:GPU:1*
_output_shapes
:*
T0
?
7My_GPU_1/gradients/My_GPU_1/add_3_grad/tuple/group_depsNoOp/^My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape1^My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape_1"/device:GPU:1
?
?My_GPU_1/gradients/My_GPU_1/add_3_grad/tuple/control_dependencyIdentity.My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape8^My_GPU_1/gradients/My_GPU_1/add_3_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape
?
AMy_GPU_1/gradients/My_GPU_1/add_3_grad/tuple/control_dependency_1Identity0My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape_18^My_GPU_1/gradients/My_GPU_1/add_3_grad/tuple/group_deps"/device:GPU:1*C
_class9
75loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape_1*
_output_shapes
:*
T0
?
7My_GPU_1/gradients/My_GPU_1/add_2_grad/tuple/group_depsNoOp@^My_GPU_1/gradients/My_GPU_1/add_3_grad/tuple/control_dependency"/device:GPU:1
?
?My_GPU_1/gradients/My_GPU_1/add_2_grad/tuple/control_dependencyIdentity?My_GPU_1/gradients/My_GPU_1/add_3_grad/tuple/control_dependency8^My_GPU_1/gradients/My_GPU_1/add_2_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: *
T0
?
AMy_GPU_1/gradients/My_GPU_1/add_2_grad/tuple/control_dependency_1Identity?My_GPU_1/gradients/My_GPU_1/add_3_grad/tuple/control_dependency8^My_GPU_1/gradients/My_GPU_1/add_2_grad/tuple/group_deps"/device:GPU:1*
T0*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape
?
,My_GPU_1/gradients/My_GPU_1/mul_2_grad/ShapeShapeMy_GPU_1/PyFunc"/device:GPU:1*
T0*#
_output_shapes
:?????????
{
.My_GPU_1/gradients/My_GPU_1/mul_2_grad/Shape_1ShapeMy_GPU_1/mul_2/y"/device:GPU:1*
T0*
_output_shapes
: 
?
<My_GPU_1/gradients/My_GPU_1/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs,My_GPU_1/gradients/My_GPU_1/mul_2_grad/Shape.My_GPU_1/gradients/My_GPU_1/mul_2_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
*My_GPU_1/gradients/My_GPU_1/mul_2_grad/MulMulAMy_GPU_1/gradients/My_GPU_1/add_3_grad/tuple/control_dependency_1My_GPU_1/mul_2/y"/device:GPU:1*
T0*
_output_shapes
:
?
*My_GPU_1/gradients/My_GPU_1/mul_2_grad/SumSum*My_GPU_1/gradients/My_GPU_1/mul_2_grad/Mul<My_GPU_1/gradients/My_GPU_1/mul_2_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
.My_GPU_1/gradients/My_GPU_1/mul_2_grad/ReshapeReshape*My_GPU_1/gradients/My_GPU_1/mul_2_grad/Sum,My_GPU_1/gradients/My_GPU_1/mul_2_grad/Shape"/device:GPU:1*
T0*
_output_shapes
:
?
,My_GPU_1/gradients/My_GPU_1/mul_2_grad/Mul_1MulMy_GPU_1/PyFuncAMy_GPU_1/gradients/My_GPU_1/add_3_grad/tuple/control_dependency_1"/device:GPU:1*
T0*
_output_shapes
:
?
,My_GPU_1/gradients/My_GPU_1/mul_2_grad/Sum_1Sum,My_GPU_1/gradients/My_GPU_1/mul_2_grad/Mul_1>My_GPU_1/gradients/My_GPU_1/mul_2_grad/BroadcastGradientArgs:1"/device:GPU:1*
T0*
_output_shapes
:
?
0My_GPU_1/gradients/My_GPU_1/mul_2_grad/Reshape_1Reshape,My_GPU_1/gradients/My_GPU_1/mul_2_grad/Sum_1.My_GPU_1/gradients/My_GPU_1/mul_2_grad/Shape_1"/device:GPU:1*
T0*
_output_shapes
: 
?
7My_GPU_1/gradients/My_GPU_1/mul_2_grad/tuple/group_depsNoOp/^My_GPU_1/gradients/My_GPU_1/mul_2_grad/Reshape1^My_GPU_1/gradients/My_GPU_1/mul_2_grad/Reshape_1"/device:GPU:1
?
?My_GPU_1/gradients/My_GPU_1/mul_2_grad/tuple/control_dependencyIdentity.My_GPU_1/gradients/My_GPU_1/mul_2_grad/Reshape8^My_GPU_1/gradients/My_GPU_1/mul_2_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/mul_2_grad/Reshape*
_output_shapes
:*
T0
?
AMy_GPU_1/gradients/My_GPU_1/mul_2_grad/tuple/control_dependency_1Identity0My_GPU_1/gradients/My_GPU_1/mul_2_grad/Reshape_18^My_GPU_1/gradients/My_GPU_1/mul_2_grad/tuple/group_deps"/device:GPU:1*
T0*
_output_shapes
: *C
_class9
75loc:@My_GPU_1/gradients/My_GPU_1/mul_2_grad/Reshape_1
?
7My_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/group_depsNoOp@^My_GPU_1/gradients/My_GPU_1/add_2_grad/tuple/control_dependency"/device:GPU:1
?
?My_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependencyIdentity?My_GPU_1/gradients/My_GPU_1/add_2_grad/tuple/control_dependency8^My_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/group_deps"/device:GPU:1*
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: 
?
AMy_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_1Identity?My_GPU_1/gradients/My_GPU_1/add_2_grad/tuple/control_dependency8^My_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
T0
?
*My_GPU_1/gradients/My_GPU_1/mul_1_grad/MulMulAMy_GPU_1/gradients/My_GPU_1/add_2_grad/tuple/control_dependency_1My_GPU_1/mul_1/y"/device:GPU:1*
_output_shapes
: *
T0
?
,My_GPU_1/gradients/My_GPU_1/mul_1_grad/Mul_1MulAMy_GPU_1/gradients/My_GPU_1/add_2_grad/tuple/control_dependency_1My_GPU_1/Mean"/device:GPU:1*
T0*
_output_shapes
: 
?
7My_GPU_1/gradients/My_GPU_1/mul_1_grad/tuple/group_depsNoOp+^My_GPU_1/gradients/My_GPU_1/mul_1_grad/Mul-^My_GPU_1/gradients/My_GPU_1/mul_1_grad/Mul_1"/device:GPU:1
?
?My_GPU_1/gradients/My_GPU_1/mul_1_grad/tuple/control_dependencyIdentity*My_GPU_1/gradients/My_GPU_1/mul_1_grad/Mul8^My_GPU_1/gradients/My_GPU_1/mul_1_grad/tuple/group_deps"/device:GPU:1*
T0*
_output_shapes
: *=
_class3
1/loc:@My_GPU_1/gradients/My_GPU_1/mul_1_grad/Mul
?
AMy_GPU_1/gradients/My_GPU_1/mul_1_grad/tuple/control_dependency_1Identity,My_GPU_1/gradients/My_GPU_1/mul_1_grad/Mul_18^My_GPU_1/gradients/My_GPU_1/mul_1_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *?
_class5
31loc:@My_GPU_1/gradients/My_GPU_1/mul_1_grad/Mul_1*
T0
?
(My_GPU_1/gradients/My_GPU_1/mul_grad/MulMul?My_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependencyMy_GPU_1/mul/y"/device:GPU:1*
_output_shapes
: *
T0
?
*My_GPU_1/gradients/My_GPU_1/mul_grad/Mul_1Mul?My_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency!My_GPU_1/mean_squared_error/value"/device:GPU:1*
T0*
_output_shapes
: 
?
5My_GPU_1/gradients/My_GPU_1/mul_grad/tuple/group_depsNoOp)^My_GPU_1/gradients/My_GPU_1/mul_grad/Mul+^My_GPU_1/gradients/My_GPU_1/mul_grad/Mul_1"/device:GPU:1
?
=My_GPU_1/gradients/My_GPU_1/mul_grad/tuple/control_dependencyIdentity(My_GPU_1/gradients/My_GPU_1/mul_grad/Mul6^My_GPU_1/gradients/My_GPU_1/mul_grad/tuple/group_deps"/device:GPU:1*
T0*;
_class1
/-loc:@My_GPU_1/gradients/My_GPU_1/mul_grad/Mul*
_output_shapes
: 
?
?My_GPU_1/gradients/My_GPU_1/mul_grad/tuple/control_dependency_1Identity*My_GPU_1/gradients/My_GPU_1/mul_grad/Mul_16^My_GPU_1/gradients/My_GPU_1/mul_grad/tuple/group_deps"/device:GPU:1*=
_class3
1/loc:@My_GPU_1/gradients/My_GPU_1/mul_grad/Mul_1*
_output_shapes
: *
T0
?
6My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/group_depsNoOpB^My_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_1"/device:GPU:1
?
>My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependencyIdentityAMy_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_17^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
T0
?
@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_1IdentityAMy_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_17^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
T0
?
@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_2IdentityAMy_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_17^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: *
T0
?
@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_3IdentityAMy_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_17^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: *
T0
?
@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_4IdentityAMy_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_17^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape
?
@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_5IdentityAMy_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_17^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
T0
?
@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_6IdentityAMy_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_17^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/group_deps"/device:GPU:1*
T0*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape
?
@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_7IdentityAMy_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_17^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
T0
?
@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_8IdentityAMy_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_17^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape
?
@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_9IdentityAMy_GPU_1/gradients/My_GPU_1/add_1_grad/tuple/control_dependency_17^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/group_deps"/device:GPU:1*
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: 
?
3My_GPU_1/gradients/My_GPU_1/Mean_grad/Reshape/shapeConst"/device:GPU:1*
dtype0*
_output_shapes
:*
valueB"      
?
-My_GPU_1/gradients/My_GPU_1/Mean_grad/ReshapeReshape?My_GPU_1/gradients/My_GPU_1/mul_1_grad/tuple/control_dependency3My_GPU_1/gradients/My_GPU_1/Mean_grad/Reshape/shape"/device:GPU:1*
T0*
_output_shapes

:
y
+My_GPU_1/gradients/My_GPU_1/Mean_grad/ShapeShapeMy_GPU_1/Square"/device:GPU:1*
_output_shapes
:*
T0
?
*My_GPU_1/gradients/My_GPU_1/Mean_grad/TileTile-My_GPU_1/gradients/My_GPU_1/Mean_grad/Reshape+My_GPU_1/gradients/My_GPU_1/Mean_grad/Shape"/device:GPU:1*
T0*'
_output_shapes
:?????????
{
-My_GPU_1/gradients/My_GPU_1/Mean_grad/Shape_1ShapeMy_GPU_1/Square"/device:GPU:1*
_output_shapes
:*
T0

-My_GPU_1/gradients/My_GPU_1/Mean_grad/Shape_2Const"/device:GPU:1*
valueB *
dtype0*
_output_shapes
: 
?
+My_GPU_1/gradients/My_GPU_1/Mean_grad/ConstConst"/device:GPU:1*
dtype0*
valueB: *
_output_shapes
:
?
*My_GPU_1/gradients/My_GPU_1/Mean_grad/ProdProd-My_GPU_1/gradients/My_GPU_1/Mean_grad/Shape_1+My_GPU_1/gradients/My_GPU_1/Mean_grad/Const"/device:GPU:1*
T0*
_output_shapes
: 
?
-My_GPU_1/gradients/My_GPU_1/Mean_grad/Const_1Const"/device:GPU:1*
_output_shapes
:*
dtype0*
valueB: 
?
,My_GPU_1/gradients/My_GPU_1/Mean_grad/Prod_1Prod-My_GPU_1/gradients/My_GPU_1/Mean_grad/Shape_2-My_GPU_1/gradients/My_GPU_1/Mean_grad/Const_1"/device:GPU:1*
T0*
_output_shapes
: 
?
/My_GPU_1/gradients/My_GPU_1/Mean_grad/Maximum/yConst"/device:GPU:1*
_output_shapes
: *
value	B :*
dtype0
?
-My_GPU_1/gradients/My_GPU_1/Mean_grad/MaximumMaximum,My_GPU_1/gradients/My_GPU_1/Mean_grad/Prod_1/My_GPU_1/gradients/My_GPU_1/Mean_grad/Maximum/y"/device:GPU:1*
_output_shapes
: *
T0
?
.My_GPU_1/gradients/My_GPU_1/Mean_grad/floordivFloorDiv*My_GPU_1/gradients/My_GPU_1/Mean_grad/Prod-My_GPU_1/gradients/My_GPU_1/Mean_grad/Maximum"/device:GPU:1*
T0*
_output_shapes
: 
?
*My_GPU_1/gradients/My_GPU_1/Mean_grad/CastCast.My_GPU_1/gradients/My_GPU_1/Mean_grad/floordiv"/device:GPU:1*
_output_shapes
: *

DstT0*

SrcT0
?
-My_GPU_1/gradients/My_GPU_1/Mean_grad/truedivRealDiv*My_GPU_1/gradients/My_GPU_1/Mean_grad/Tile*My_GPU_1/gradients/My_GPU_1/Mean_grad/Cast"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
My_GPU_1/gradients/AddNAddNFMy_GPU_1/gradients/My_GPU_1/total_loss_1_grad/tuple/control_dependency=My_GPU_1/gradients/My_GPU_1/mul_grad/tuple/control_dependency"/device:GPU:1**
_class 
loc:@My_GPU_1/gradients/Fill*
T0*
N*
_output_shapes
: 
?
?My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/ShapeConst"/device:GPU:1*
valueB *
_output_shapes
: *
dtype0
?
AMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Shape_1Const"/device:GPU:1*
_output_shapes
: *
valueB *
dtype0
?
OMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/BroadcastGradientArgsBroadcastGradientArgs?My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/ShapeAMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
DMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/div_no_nanDivNoNanMy_GPU_1/gradients/AddN'My_GPU_1/mean_squared_error/num_present"/device:GPU:1*
T0*
_output_shapes
: 
?
=My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/SumSumDMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/div_no_nanOMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
: 
?
AMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/ReshapeReshape=My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Sum?My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Shape"/device:GPU:1*
T0*
_output_shapes
: 
?
=My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/NegNeg!My_GPU_1/mean_squared_error/Sum_1"/device:GPU:1*
T0*
_output_shapes
: 
?
FMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/div_no_nan_1DivNoNan=My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Neg'My_GPU_1/mean_squared_error/num_present"/device:GPU:1*
_output_shapes
: *
T0
?
FMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/div_no_nan_2DivNoNanFMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/div_no_nan_1'My_GPU_1/mean_squared_error/num_present"/device:GPU:1*
_output_shapes
: *
T0
?
=My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/mulMulMy_GPU_1/gradients/AddNFMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/div_no_nan_2"/device:GPU:1*
T0*
_output_shapes
: 
?
?My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Sum_1Sum=My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/mulQMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/BroadcastGradientArgs:1"/device:GPU:1*
T0*
_output_shapes
: 
?
CMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Reshape_1Reshape?My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Sum_1AMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Shape_1"/device:GPU:1*
_output_shapes
: *
T0
?
JMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/tuple/group_depsNoOpB^My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/ReshapeD^My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Reshape_1"/device:GPU:1
?
RMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/tuple/control_dependencyIdentityAMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/ReshapeK^My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/tuple/group_deps"/device:GPU:1*
T0*T
_classJ
HFloc:@My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Reshape*
_output_shapes
: 
?
TMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/tuple/control_dependency_1IdentityCMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Reshape_1K^My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/tuple/group_deps"/device:GPU:1*
T0*V
_classL
JHloc:@My_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/Reshape_1*
_output_shapes
: 
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/add_grad/tuple/group_depsNoOp?^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/add_grad/tuple/control_dependencyIdentity>My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
T0*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/add_grad/tuple/control_dependency_1Identity>My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/add_grad/tuple/group_depsNoOpA^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_1"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/add_grad/tuple/control_dependencyIdentity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_1^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
T0
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/add_grad/tuple/control_dependency_1Identity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_1^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
T0*
_output_shapes
: 
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/add_grad/tuple/group_depsNoOpA^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_2"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/add_grad/tuple/control_dependencyIdentity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_2^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
T0*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/add_grad/tuple/control_dependency_1Identity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_2^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/add_grad/tuple/group_depsNoOpA^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_3"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/add_grad/tuple/control_dependencyIdentity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_3^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
T0*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/add_grad/tuple/control_dependency_1Identity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_3^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
T0
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/add_grad/tuple/group_depsNoOpA^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_4"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/add_grad/tuple/control_dependencyIdentity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_4^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: *
T0
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/add_grad/tuple/control_dependency_1Identity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_4^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
T0*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/add_grad/tuple/group_depsNoOpA^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_5"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/add_grad/tuple/control_dependencyIdentity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_5^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
T0*
_output_shapes
: 
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/add_grad/tuple/control_dependency_1Identity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_5^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: 
?
dMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/add_grad/tuple/group_depsNoOpA^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_6"/device:GPU:1
?
lMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/add_grad/tuple/control_dependencyIdentity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_6e^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: 
?
nMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/add_grad/tuple/control_dependency_1Identity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_6e^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: 
?
NMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/add_grad/tuple/group_depsNoOpA^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_7"/device:GPU:1
?
VMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/add_grad/tuple/control_dependencyIdentity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_7O^My_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: *
T0
?
XMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/add_grad/tuple/control_dependency_1Identity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_7O^My_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: *
T0
?
`My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/add_grad/tuple/group_depsNoOpA^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_8"/device:GPU:1
?
hMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/add_grad/tuple/control_dependencyIdentity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_8a^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: *
T0
?
jMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/add_grad/tuple/control_dependency_1Identity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_8a^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: 
?
VMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/add_grad/tuple/group_depsNoOpA^My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_9"/device:GPU:1
?
^My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/add_grad/tuple/control_dependencyIdentity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_9W^My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
T0
?
`My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/add_grad/tuple/control_dependency_1Identity@My_GPU_1/gradients/My_GPU_1/AddN_grad/tuple/control_dependency_9W^My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/add_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_3_grad/Reshape*
_output_shapes
: *
T0
?
-My_GPU_1/gradients/My_GPU_1/Square_grad/ConstConst.^My_GPU_1/gradients/My_GPU_1/Mean_grad/truediv"/device:GPU:1*
dtype0*
valueB
 *   @*
_output_shapes
: 
?
+My_GPU_1/gradients/My_GPU_1/Square_grad/MulMulMy_GPU_1/sub-My_GPU_1/gradients/My_GPU_1/Square_grad/Const"/device:GPU:1*
T0*'
_output_shapes
:?????????
?
-My_GPU_1/gradients/My_GPU_1/Square_grad/Mul_1Mul-My_GPU_1/gradients/My_GPU_1/Mean_grad/truediv+My_GPU_1/gradients/My_GPU_1/Square_grad/Mul"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
GMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_1_grad/Reshape/shapeConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB 
?
AMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_1_grad/ReshapeReshapeRMy_GPU_1/gradients/My_GPU_1/mean_squared_error/value_grad/tuple/control_dependencyGMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_1_grad/Reshape/shape"/device:GPU:1*
T0*
_output_shapes
: 
?
?My_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_1_grad/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB *
dtype0
?
>My_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_1_grad/TileTileAMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_1_grad/Reshape?My_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_1_grad/Const"/device:GPU:1*
_output_shapes
: *
T0
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/MulMulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/add_grad/tuple/control_dependency_14My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Sum"/device:GPU:1*
T0*
_output_shapes
: 
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/Mul_1MulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/add_grad/tuple/control_dependency_16My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul/x"/device:GPU:1*
_output_shapes
: *
T0
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/tuple/group_depsNoOpQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/MulS^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/Mul_1"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/Mul^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/Mul*
T0*
_output_shapes
: 
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/Mul_1^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
T0*e
_class[
YWloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/Mul_1*
_output_shapes
: 
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/MulMulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/add_grad/tuple/control_dependency_14My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Sum"/device:GPU:1*
T0*
_output_shapes
: 
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/Mul_1MulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/add_grad/tuple/control_dependency_16My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul/x"/device:GPU:1*
T0*
_output_shapes
: 
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/tuple/group_depsNoOpQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/MulS^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/Mul_1"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/Mul^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/Mul
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/Mul_1^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*e
_class[
YWloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/Mul_1*
T0*
_output_shapes
: 
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/MulMulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/add_grad/tuple/control_dependency_14My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Sum"/device:GPU:1*
_output_shapes
: *
T0
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/Mul_1MulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/add_grad/tuple/control_dependency_16My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul/x"/device:GPU:1*
T0*
_output_shapes
: 
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/tuple/group_depsNoOpQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/MulS^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/Mul_1"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/Mul^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
T0*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/Mul*
_output_shapes
: 
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/Mul_1^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*e
_class[
YWloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/Mul_1*
_output_shapes
: *
T0
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/MulMulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/add_grad/tuple/control_dependency_14My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Sum"/device:GPU:1*
T0*
_output_shapes
: 
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/Mul_1MulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/add_grad/tuple/control_dependency_16My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul/x"/device:GPU:1*
_output_shapes
: *
T0
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/tuple/group_depsNoOpQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/MulS^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/Mul_1"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/Mul^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/Mul
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/Mul_1^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*e
_class[
YWloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/Mul_1*
T0*
_output_shapes
: 
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/MulMulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/add_grad/tuple/control_dependency_14My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Sum"/device:GPU:1*
T0*
_output_shapes
: 
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/Mul_1MulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/add_grad/tuple/control_dependency_16My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul/x"/device:GPU:1*
T0*
_output_shapes
: 
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/tuple/group_depsNoOpQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/MulS^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/Mul_1"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/Mul^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/Mul
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/Mul_1^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*e
_class[
YWloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/Mul_1
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/MulMulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/add_grad/tuple/control_dependency_14My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Sum"/device:GPU:1*
_output_shapes
: *
T0
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/Mul_1MulgMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/add_grad/tuple/control_dependency_16My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul/x"/device:GPU:1*
T0*
_output_shapes
: 
?
]My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/tuple/group_depsNoOpQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/MulS^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/Mul_1"/device:GPU:1
?
eMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/Mul^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/Mul
?
gMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/Mul_1^^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*e
_class[
YWloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/Mul_1*
T0*
_output_shapes
: 
?
WMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/MulMulnMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/add_grad/tuple/control_dependency_1;My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Sum"/device:GPU:1*
_output_shapes
: *
T0
?
YMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/Mul_1MulnMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/add_grad/tuple/control_dependency_1=My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul/x"/device:GPU:1*
_output_shapes
: *
T0
?
dMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/tuple/group_depsNoOpX^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/MulZ^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/Mul_1"/device:GPU:1
?
lMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityWMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/Mule^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
T0*
_output_shapes
: *j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/Mul
?
nMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityYMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/Mul_1e^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *l
_classb
`^loc:@My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/Mul_1*
T0
?
AMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/MulMulXMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/add_grad/tuple/control_dependency_1%My_GPU_1/dense/kernel/Regularizer/Sum"/device:GPU:1*
T0*
_output_shapes
: 
?
CMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/Mul_1MulXMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/add_grad/tuple/control_dependency_1'My_GPU_1/dense/kernel/Regularizer/mul/x"/device:GPU:1*
T0*
_output_shapes
: 
?
NMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/tuple/group_depsNoOpB^My_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/MulD^My_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/Mul_1"/device:GPU:1
?
VMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityAMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/MulO^My_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
T0*T
_classJ
HFloc:@My_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/Mul*
_output_shapes
: 
?
XMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityCMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/Mul_1O^My_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*V
_classL
JHloc:@My_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/Mul_1*
_output_shapes
: *
T0
?
SMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/MulMuljMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/add_grad/tuple/control_dependency_17My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Sum"/device:GPU:1*
_output_shapes
: *
T0
?
UMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/Mul_1MuljMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/add_grad/tuple/control_dependency_19My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul/x"/device:GPU:1*
_output_shapes
: *
T0
?
`My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/tuple/group_depsNoOpT^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/MulV^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/Mul_1"/device:GPU:1
?
hMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentitySMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/Mula^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*f
_class\
ZXloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/Mul
?
jMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityUMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/Mul_1a^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
T0*h
_class^
\Zloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/Mul_1*
_output_shapes
: 
?
IMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/MulMul`My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/add_grad/tuple/control_dependency_1-My_GPU_1/Output_/dense/kernel/Regularizer/Sum"/device:GPU:1*
T0*
_output_shapes
: 
?
KMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/Mul_1Mul`My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/add_grad/tuple/control_dependency_1/My_GPU_1/Output_/dense/kernel/Regularizer/mul/x"/device:GPU:1*
T0*
_output_shapes
: 
?
VMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/tuple/group_depsNoOpJ^My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/MulL^My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/Mul_1"/device:GPU:1
?
^My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/tuple/control_dependencyIdentityIMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/MulW^My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/Mul
?
`My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/tuple/control_dependency_1IdentityKMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/Mul_1W^My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*^
_classT
RPloc:@My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/Mul_1
y
*My_GPU_1/gradients/My_GPU_1/sub_grad/ShapeShapeMy_GPU_1/Reshape"/device:GPU:1*
T0*
_output_shapes
:
?
,My_GPU_1/gradients/My_GPU_1/sub_grad/Shape_1Shape,My_GPU_1/Reconstruction_Output/dense/BiasAdd"/device:GPU:1*
_output_shapes
:*
T0
?
:My_GPU_1/gradients/My_GPU_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*My_GPU_1/gradients/My_GPU_1/sub_grad/Shape,My_GPU_1/gradients/My_GPU_1/sub_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
(My_GPU_1/gradients/My_GPU_1/sub_grad/SumSum-My_GPU_1/gradients/My_GPU_1/Square_grad/Mul_1:My_GPU_1/gradients/My_GPU_1/sub_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
,My_GPU_1/gradients/My_GPU_1/sub_grad/ReshapeReshape(My_GPU_1/gradients/My_GPU_1/sub_grad/Sum*My_GPU_1/gradients/My_GPU_1/sub_grad/Shape"/device:GPU:1*0
_output_shapes
:??????????????????*
T0
?
(My_GPU_1/gradients/My_GPU_1/sub_grad/NegNeg-My_GPU_1/gradients/My_GPU_1/Square_grad/Mul_1"/device:GPU:1*
T0*'
_output_shapes
:?????????
?
*My_GPU_1/gradients/My_GPU_1/sub_grad/Sum_1Sum(My_GPU_1/gradients/My_GPU_1/sub_grad/Neg<My_GPU_1/gradients/My_GPU_1/sub_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
.My_GPU_1/gradients/My_GPU_1/sub_grad/Reshape_1Reshape*My_GPU_1/gradients/My_GPU_1/sub_grad/Sum_1,My_GPU_1/gradients/My_GPU_1/sub_grad/Shape_1"/device:GPU:1*
T0*'
_output_shapes
:?????????
?
5My_GPU_1/gradients/My_GPU_1/sub_grad/tuple/group_depsNoOp-^My_GPU_1/gradients/My_GPU_1/sub_grad/Reshape/^My_GPU_1/gradients/My_GPU_1/sub_grad/Reshape_1"/device:GPU:1
?
=My_GPU_1/gradients/My_GPU_1/sub_grad/tuple/control_dependencyIdentity,My_GPU_1/gradients/My_GPU_1/sub_grad/Reshape6^My_GPU_1/gradients/My_GPU_1/sub_grad/tuple/group_deps"/device:GPU:1*0
_output_shapes
:??????????????????*?
_class5
31loc:@My_GPU_1/gradients/My_GPU_1/sub_grad/Reshape*
T0
?
?My_GPU_1/gradients/My_GPU_1/sub_grad/tuple/control_dependency_1Identity.My_GPU_1/gradients/My_GPU_1/sub_grad/Reshape_16^My_GPU_1/gradients/My_GPU_1/sub_grad/tuple/group_deps"/device:GPU:1*
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/sub_grad/Reshape_1*'
_output_shapes
:?????????
?
EMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_grad/Reshape/shapeConst"/device:GPU:1*
valueB:*
_output_shapes
:*
dtype0
?
?My_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_grad/ReshapeReshape>My_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_1_grad/TileEMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_grad/Reshape/shape"/device:GPU:1*
_output_shapes
:*
T0
?
=My_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_grad/ShapeShapeMy_GPU_1/mean_squared_error/Mul"/device:GPU:1*
_output_shapes
:*
T0
?
<My_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_grad/TileTile?My_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_grad/Reshape=My_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_grad/Shape"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Sum_grad/Reshape/shapeConst"/device:GPU:1*
_output_shapes
:*
dtype0*!
valueB"         
?
TMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Sum_grad/ReshapeReshapegMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/mul_grad/tuple/control_dependency_1ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Sum_grad/Reshape/shape"/device:GPU:1*
T0*"
_output_shapes
:
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Sum_grad/ConstConst"/device:GPU:1*!
valueB"         *
dtype0*
_output_shapes
:
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Sum_grad/TileTileTMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Sum_grad/ReshapeRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Sum_grad/Const"/device:GPU:1*#
_output_shapes
:?*
T0
?
ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Sum_grad/Reshape/shapeConst"/device:GPU:1*!
valueB"         *
_output_shapes
:*
dtype0
?
TMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Sum_grad/ReshapeReshapegMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/mul_grad/tuple/control_dependency_1ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Sum_grad/Reshape/shape"/device:GPU:1*
T0*"
_output_shapes
:
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Sum_grad/ConstConst"/device:GPU:1*
dtype0*!
valueB"         *
_output_shapes
:
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Sum_grad/TileTileTMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Sum_grad/ReshapeRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Sum_grad/Const"/device:GPU:1*
T0*$
_output_shapes
:??
?
ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Sum_grad/Reshape/shapeConst"/device:GPU:1*!
valueB"         *
dtype0*
_output_shapes
:
?
TMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Sum_grad/ReshapeReshapegMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/mul_grad/tuple/control_dependency_1ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Sum_grad/Reshape/shape"/device:GPU:1*
T0*"
_output_shapes
:
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Sum_grad/ConstConst"/device:GPU:1*!
valueB"         *
dtype0*
_output_shapes
:
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Sum_grad/TileTileTMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Sum_grad/ReshapeRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Sum_grad/Const"/device:GPU:1*$
_output_shapes
:??*
T0
?
ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Sum_grad/Reshape/shapeConst"/device:GPU:1*
dtype0*!
valueB"         *
_output_shapes
:
?
TMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Sum_grad/ReshapeReshapegMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/mul_grad/tuple/control_dependency_1ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Sum_grad/Reshape/shape"/device:GPU:1*
T0*"
_output_shapes
:
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Sum_grad/ConstConst"/device:GPU:1*
_output_shapes
:*
dtype0*!
valueB"         
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Sum_grad/TileTileTMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Sum_grad/ReshapeRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Sum_grad/Const"/device:GPU:1*$
_output_shapes
:??*
T0
?
ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Sum_grad/Reshape/shapeConst"/device:GPU:1*
dtype0*
_output_shapes
:*!
valueB"         
?
TMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Sum_grad/ReshapeReshapegMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/mul_grad/tuple/control_dependency_1ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Sum_grad/Reshape/shape"/device:GPU:1*"
_output_shapes
:*
T0
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Sum_grad/ConstConst"/device:GPU:1*
dtype0*
_output_shapes
:*!
valueB"         
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Sum_grad/TileTileTMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Sum_grad/ReshapeRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Sum_grad/Const"/device:GPU:1*
T0*$
_output_shapes
:??
?
ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Sum_grad/Reshape/shapeConst"/device:GPU:1*!
valueB"         *
_output_shapes
:*
dtype0
?
TMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Sum_grad/ReshapeReshapegMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/mul_grad/tuple/control_dependency_1ZMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Sum_grad/Reshape/shape"/device:GPU:1*
T0*"
_output_shapes
:
?
RMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Sum_grad/ConstConst"/device:GPU:1*
dtype0*
_output_shapes
:*!
valueB"         
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Sum_grad/TileTileTMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Sum_grad/ReshapeRMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Sum_grad/Const"/device:GPU:1*
T0*$
_output_shapes
:??
?
aMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Sum_grad/Reshape/shapeConst"/device:GPU:1*
_output_shapes
:*
dtype0*
valueB"      
?
[My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Sum_grad/ReshapeReshapenMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/mul_grad/tuple/control_dependency_1aMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Sum_grad/Reshape/shape"/device:GPU:1*
T0*
_output_shapes

:
?
YMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Sum_grad/ConstConst"/device:GPU:1*
dtype0*
valueB"      *
_output_shapes
:
?
XMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Sum_grad/TileTile[My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Sum_grad/ReshapeYMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Sum_grad/Const"/device:GPU:1*
T0*
_output_shapes
:	?
?
KMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Sum_grad/Reshape/shapeConst"/device:GPU:1*
valueB"      *
_output_shapes
:*
dtype0
?
EMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Sum_grad/ReshapeReshapeXMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/mul_grad/tuple/control_dependency_1KMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Sum_grad/Reshape/shape"/device:GPU:1*
_output_shapes

:*
T0
?
CMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Sum_grad/ConstConst"/device:GPU:1*
_output_shapes
:*
valueB"       *
dtype0
?
BMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Sum_grad/TileTileEMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Sum_grad/ReshapeCMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Sum_grad/Const"/device:GPU:1*
T0*
_output_shapes
:	? 
?
]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Sum_grad/Reshape/shapeConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
?
WMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Sum_grad/ReshapeReshapejMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/mul_grad/tuple/control_dependency_1]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Sum_grad/Reshape/shape"/device:GPU:1*
T0*
_output_shapes

:
?
UMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Sum_grad/ConstConst"/device:GPU:1*
_output_shapes
:*
dtype0*
valueB"        
?
TMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Sum_grad/TileTileWMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Sum_grad/ReshapeUMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Sum_grad/Const"/device:GPU:1*
_output_shapes

:  *
T0
?
SMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Sum_grad/Reshape/shapeConst"/device:GPU:1*
_output_shapes
:*
dtype0*
valueB"      
?
MMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Sum_grad/ReshapeReshape`My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/mul_grad/tuple/control_dependency_1SMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Sum_grad/Reshape/shape"/device:GPU:1*
_output_shapes

:*
T0
?
KMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Sum_grad/ConstConst"/device:GPU:1*
valueB"       *
_output_shapes
:*
dtype0
?
JMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Sum_grad/TileTileMMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Sum_grad/ReshapeKMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Sum_grad/Const"/device:GPU:1*
T0*
_output_shapes

: 
?
PMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/BiasAddGradBiasAddGrad?My_GPU_1/gradients/My_GPU_1/sub_grad/tuple/control_dependency_1"/device:GPU:1*
T0*
_output_shapes
:
?
UMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/tuple/group_depsNoOpQ^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/BiasAddGrad@^My_GPU_1/gradients/My_GPU_1/sub_grad/tuple/control_dependency_1"/device:GPU:1
?
]My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/tuple/control_dependencyIdentity?My_GPU_1/gradients/My_GPU_1/sub_grad/tuple/control_dependency_1V^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:?????????*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/sub_grad/Reshape_1*
T0
?
_My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/tuple/control_dependency_1IdentityPMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/BiasAddGradV^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
:*
T0*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/BiasAddGrad
?
=My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/ShapeShape-My_GPU_1/mean_squared_error/SquaredDifference"/device:GPU:1*
_output_shapes
:*
T0
?
?My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Shape_1Shape"My_GPU_1/mean_squared_error/Cast/x"/device:GPU:1*
_output_shapes
: *
T0
?
MMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs=My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Shape?My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
;My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/MulMul<My_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_grad/Tile"My_GPU_1/mean_squared_error/Cast/x"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
;My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/SumSum;My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/MulMMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
?My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/ReshapeReshape;My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Sum=My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Shape"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
=My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Mul_1Mul-My_GPU_1/mean_squared_error/SquaredDifference<My_GPU_1/gradients/My_GPU_1/mean_squared_error/Sum_grad/Tile"/device:GPU:1*#
_output_shapes
:?????????*
T0
?
=My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Sum_1Sum=My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Mul_1OMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/BroadcastGradientArgs:1"/device:GPU:1*
T0*
_output_shapes
:
?
AMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Reshape_1Reshape=My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Sum_1?My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Shape_1"/device:GPU:1*
T0*
_output_shapes
: 
?
HMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/tuple/group_depsNoOp@^My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/ReshapeB^My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Reshape_1"/device:GPU:1
?
PMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity?My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/ReshapeI^My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/tuple/group_deps"/device:GPU:1*
T0*R
_classH
FDloc:@My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Reshape*#
_output_shapes
:?????????
?
RMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/tuple/control_dependency_1IdentityAMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Reshape_1I^My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *T
_classJ
HFloc:@My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/Reshape_1*
T0
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Square_grad/ConstConstR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Sum_grad/Tile"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB
 *   @
?
SMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Square_grad/MulMul CCN_1Conv_x0/convA10/kernel/readUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Square_grad/Const"/device:GPU:1*#
_output_shapes
:?*
T0
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Square_grad/Mul_1MulQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Sum_grad/TileSMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Square_grad/Mul"/device:GPU:1*
T0*#
_output_shapes
:?
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Square_grad/ConstConstR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Sum_grad/Tile"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *   @
?
SMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Square_grad/MulMul CCN_1Conv_x0/convB10/kernel/readUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Square_grad/Const"/device:GPU:1*$
_output_shapes
:??*
T0
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Square_grad/Mul_1MulQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Sum_grad/TileSMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Square_grad/Mul"/device:GPU:1*$
_output_shapes
:??*
T0
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Square_grad/ConstConstR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Sum_grad/Tile"/device:GPU:1*
valueB
 *   @*
_output_shapes
: *
dtype0
?
SMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Square_grad/MulMul CCN_1Conv_x0/convB20/kernel/readUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Square_grad/Const"/device:GPU:1*$
_output_shapes
:??*
T0
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Square_grad/Mul_1MulQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Sum_grad/TileSMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Square_grad/Mul"/device:GPU:1*$
_output_shapes
:??*
T0
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Square_grad/ConstConstR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Sum_grad/Tile"/device:GPU:1*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
SMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Square_grad/MulMul CCN_1Conv_x0/convA11/kernel/readUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Square_grad/Const"/device:GPU:1*$
_output_shapes
:??*
T0
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Square_grad/Mul_1MulQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Sum_grad/TileSMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Square_grad/Mul"/device:GPU:1*
T0*$
_output_shapes
:??
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Square_grad/ConstConstR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Sum_grad/Tile"/device:GPU:1*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
SMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Square_grad/MulMul CCN_1Conv_x0/convB11/kernel/readUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Square_grad/Const"/device:GPU:1*$
_output_shapes
:??*
T0
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Square_grad/Mul_1MulQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Sum_grad/TileSMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Square_grad/Mul"/device:GPU:1*$
_output_shapes
:??*
T0
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Square_grad/ConstConstR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Sum_grad/Tile"/device:GPU:1*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
SMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Square_grad/MulMul CCN_1Conv_x0/convB21/kernel/readUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Square_grad/Const"/device:GPU:1*$
_output_shapes
:??*
T0
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Square_grad/Mul_1MulQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Sum_grad/TileSMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Square_grad/Mul"/device:GPU:1*
T0*$
_output_shapes
:??
?
\My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Square_grad/ConstConstY^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Sum_grad/Tile"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB
 *   @
?
ZMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Square_grad/MulMul'Reconstruction_Output/dense/kernel/read\My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Square_grad/Const"/device:GPU:1*
T0*
_output_shapes
:	?
?
\My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Square_grad/Mul_1MulXMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Sum_grad/TileZMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Square_grad/Mul"/device:GPU:1*
_output_shapes
:	?*
T0
?
FMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Square_grad/ConstConstC^My_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Sum_grad/Tile"/device:GPU:1*
dtype0*
valueB
 *   @*
_output_shapes
: 
?
DMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Square_grad/MulMuldense/kernel/readFMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Square_grad/Const"/device:GPU:1*
T0*
_output_shapes
:	? 
?
FMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Square_grad/Mul_1MulBMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Sum_grad/TileDMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Square_grad/Mul"/device:GPU:1*
T0*
_output_shapes
:	? 
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Square_grad/ConstConstU^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Sum_grad/Tile"/device:GPU:1*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Square_grad/MulMul#FCU_muiltDense_x0/dense/kernel/readXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Square_grad/Const"/device:GPU:1*
T0*
_output_shapes

:  
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Square_grad/Mul_1MulTMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Sum_grad/TileVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Square_grad/Mul"/device:GPU:1*
T0*
_output_shapes

:  
?
NMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Square_grad/ConstConstK^My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Sum_grad/Tile"/device:GPU:1*
valueB
 *   @*
_output_shapes
: *
dtype0
?
LMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Square_grad/MulMulOutput_/dense/kernel/readNMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Square_grad/Const"/device:GPU:1*
_output_shapes

: *
T0
?
NMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Square_grad/Mul_1MulJMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Sum_grad/TileLMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Square_grad/Mul"/device:GPU:1*
_output_shapes

: *
T0
?
JMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/MatMulMatMul]My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/tuple/control_dependency'Reconstruction_Output/dense/kernel/read"/device:GPU:1*
T0*
transpose_b(*(
_output_shapes
:??????????
?
LMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/MatMul_1MatMul#My_GPU_1/Conv_out__/Conv_out__/Relu]My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/tuple/control_dependency"/device:GPU:1*
transpose_a(*
_output_shapes
:	?*
T0
?
TMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/tuple/group_depsNoOpK^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/MatMulM^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/MatMul_1"/device:GPU:1
?
\My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/tuple/control_dependencyIdentityJMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/MatMulU^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/tuple/group_deps"/device:GPU:1*]
_classS
QOloc:@My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/MatMul*(
_output_shapes
:??????????*
T0
?
^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/tuple/control_dependency_1IdentityLMy_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/MatMul_1U^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/tuple/group_deps"/device:GPU:1*
T0*_
_classU
SQloc:@My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/MatMul_1*
_output_shapes
:	?
?
LMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/scalarConstQ^My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/tuple/control_dependency"/device:GPU:1*
valueB
 *   @*
dtype0*
_output_shapes
: 
?
IMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/MulMulLMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/scalarPMy_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/tuple/control_dependency"/device:GPU:1*#
_output_shapes
:?????????*
T0
?
IMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/subSubMy_GPU_1/strided_slice_1My_GPU_1/strided_sliceQ^My_GPU_1/gradients/My_GPU_1/mean_squared_error/Mul_grad/tuple/control_dependency"/device:GPU:1*#
_output_shapes
:?????????*
T0
?
KMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/mul_1MulIMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/MulIMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/sub"/device:GPU:1*#
_output_shapes
:?????????*
T0
?
KMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/ShapeShapeMy_GPU_1/strided_slice_1"/device:GPU:1*
T0*
_output_shapes
:
?
MMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/Shape_1ShapeMy_GPU_1/strided_slice"/device:GPU:1*
T0*
_output_shapes
:
?
[My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsKMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/ShapeMMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
IMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/SumSumKMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/mul_1[My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
MMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/ReshapeReshapeIMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/SumKMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/Shape"/device:GPU:1*#
_output_shapes
:?????????*
T0
?
KMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/Sum_1SumKMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/mul_1]My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
OMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/Reshape_1ReshapeKMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/Sum_1MMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/Shape_1"/device:GPU:1*#
_output_shapes
:?????????*
T0
?
IMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/NegNegOMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/Reshape_1"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
VMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOpJ^My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/NegN^My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/Reshape"/device:GPU:1
?
^My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentityMMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/ReshapeW^My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/tuple/group_deps"/device:GPU:1*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/Reshape*#
_output_shapes
:?????????*
T0
?
`My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1IdentityIMy_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/NegW^My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/tuple/group_deps"/device:GPU:1*#
_output_shapes
:?????????*
T0*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/Neg
?
My_GPU_1/gradients/AddN_1AddN\My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Square_grad/Mul_1^My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/tuple/control_dependency_1"/device:GPU:1*
T0*
_output_shapes
:	?*
N*o
_classe
caloc:@My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/Square_grad/Mul_1
?
6My_GPU_1/gradients/My_GPU_1/strided_slice_1_grad/ShapeShapeMy_GPU_1/Output_/dense/BiasAdd"/device:GPU:1*
T0*
_output_shapes
:
?
AMy_GPU_1/gradients/My_GPU_1/strided_slice_1_grad/StridedSliceGradStridedSliceGrad6My_GPU_1/gradients/My_GPU_1/strided_slice_1_grad/ShapeMy_GPU_1/strided_slice_1/stack My_GPU_1/strided_slice_1/stack_1 My_GPU_1/strided_slice_1/stack_2^My_GPU_1/gradients/My_GPU_1/mean_squared_error/SquaredDifference_grad/tuple/control_dependency"/device:GPU:1*'
_output_shapes
:?????????*
end_mask*
Index0*
T0*

begin_mask*
shrink_axis_mask
?
BMy_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/BiasAddGradBiasAddGradAMy_GPU_1/gradients/My_GPU_1/strided_slice_1_grad/StridedSliceGrad"/device:GPU:1*
_output_shapes
:*
T0
?
GMy_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/tuple/group_depsNoOpC^My_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/BiasAddGradB^My_GPU_1/gradients/My_GPU_1/strided_slice_1_grad/StridedSliceGrad"/device:GPU:1
?
OMy_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/tuple/control_dependencyIdentityAMy_GPU_1/gradients/My_GPU_1/strided_slice_1_grad/StridedSliceGradH^My_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/tuple/group_deps"/device:GPU:1*
T0*T
_classJ
HFloc:@My_GPU_1/gradients/My_GPU_1/strided_slice_1_grad/StridedSliceGrad*'
_output_shapes
:?????????
?
QMy_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/tuple/control_dependency_1IdentityBMy_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/BiasAddGradH^My_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/tuple/group_deps"/device:GPU:1*
T0*U
_classK
IGloc:@My_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
?
<My_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/MatMulMatMulOMy_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/tuple/control_dependencyOutput_/dense/kernel/read"/device:GPU:1*'
_output_shapes
:????????? *
T0*
transpose_b(
?
>My_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/MatMul_1MatMulMy_GPU_1/addOMy_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/tuple/control_dependency"/device:GPU:1*
transpose_a(*
T0*
_output_shapes

: 
?
FMy_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/tuple/group_depsNoOp=^My_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/MatMul?^My_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/MatMul_1"/device:GPU:1
?
NMy_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/tuple/control_dependencyIdentity<My_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/MatMulG^My_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:????????? *O
_classE
CAloc:@My_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/MatMul*
T0
?
PMy_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/tuple/control_dependency_1Identity>My_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/MatMul_1G^My_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes

: *Q
_classG
ECloc:@My_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/MatMul_1*
T0
?
*My_GPU_1/gradients/My_GPU_1/add_grad/ShapeShape(My_GPU_1/FCU_muiltDense_x0/dropout/mul_1"/device:GPU:1*
_output_shapes
:*
T0
?
,My_GPU_1/gradients/My_GPU_1/add_grad/Shape_1ShapeMy_GPU_1/dense/BiasAdd"/device:GPU:1*
T0*
_output_shapes
:
?
:My_GPU_1/gradients/My_GPU_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs*My_GPU_1/gradients/My_GPU_1/add_grad/Shape,My_GPU_1/gradients/My_GPU_1/add_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
(My_GPU_1/gradients/My_GPU_1/add_grad/SumSumNMy_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/tuple/control_dependency:My_GPU_1/gradients/My_GPU_1/add_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
,My_GPU_1/gradients/My_GPU_1/add_grad/ReshapeReshape(My_GPU_1/gradients/My_GPU_1/add_grad/Sum*My_GPU_1/gradients/My_GPU_1/add_grad/Shape"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
*My_GPU_1/gradients/My_GPU_1/add_grad/Sum_1SumNMy_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/tuple/control_dependency<My_GPU_1/gradients/My_GPU_1/add_grad/BroadcastGradientArgs:1"/device:GPU:1*
T0*
_output_shapes
:
?
.My_GPU_1/gradients/My_GPU_1/add_grad/Reshape_1Reshape*My_GPU_1/gradients/My_GPU_1/add_grad/Sum_1,My_GPU_1/gradients/My_GPU_1/add_grad/Shape_1"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
5My_GPU_1/gradients/My_GPU_1/add_grad/tuple/group_depsNoOp-^My_GPU_1/gradients/My_GPU_1/add_grad/Reshape/^My_GPU_1/gradients/My_GPU_1/add_grad/Reshape_1"/device:GPU:1
?
=My_GPU_1/gradients/My_GPU_1/add_grad/tuple/control_dependencyIdentity,My_GPU_1/gradients/My_GPU_1/add_grad/Reshape6^My_GPU_1/gradients/My_GPU_1/add_grad/tuple/group_deps"/device:GPU:1*
T0*?
_class5
31loc:@My_GPU_1/gradients/My_GPU_1/add_grad/Reshape*'
_output_shapes
:????????? 
?
?My_GPU_1/gradients/My_GPU_1/add_grad/tuple/control_dependency_1Identity.My_GPU_1/gradients/My_GPU_1/add_grad/Reshape_16^My_GPU_1/gradients/My_GPU_1/add_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:????????? *
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_grad/Reshape_1
?
My_GPU_1/gradients/AddN_2AddNNMy_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Square_grad/Mul_1PMy_GPU_1/gradients/My_GPU_1/Output_/dense/MatMul_grad/tuple/control_dependency_1"/device:GPU:1*
T0*
_output_shapes

: *a
_classW
USloc:@My_GPU_1/gradients/My_GPU_1/Output_/dense/kernel/Regularizer/Square_grad/Mul_1*
N
?
FMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/ShapeShape&My_GPU_1/FCU_muiltDense_x0/dropout/mul"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
HMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Shape_1Shape'My_GPU_1/FCU_muiltDense_x0/dropout/Cast"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsFMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/ShapeHMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
DMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/MulMul=My_GPU_1/gradients/My_GPU_1/add_grad/tuple/control_dependency'My_GPU_1/FCU_muiltDense_x0/dropout/Cast"/device:GPU:1*
T0*
_output_shapes
:
?
DMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/SumSumDMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/MulVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
HMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/ReshapeReshapeDMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/SumFMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Shape"/device:GPU:1*
_output_shapes
:*
T0
?
FMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Mul_1Mul&My_GPU_1/FCU_muiltDense_x0/dropout/mul=My_GPU_1/gradients/My_GPU_1/add_grad/tuple/control_dependency"/device:GPU:1*
T0*
_output_shapes
:
?
FMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Sum_1SumFMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Mul_1XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
JMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Reshape_1ReshapeFMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Sum_1HMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Shape_1"/device:GPU:1*
T0*
_output_shapes
:
?
QMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/tuple/group_depsNoOpI^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/ReshapeK^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Reshape_1"/device:GPU:1
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/tuple/control_dependencyIdentityHMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/ReshapeR^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/tuple/group_deps"/device:GPU:1*[
_classQ
OMloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Reshape*
_output_shapes
:*
T0
?
[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/tuple/control_dependency_1IdentityJMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Reshape_1R^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/tuple/group_deps"/device:GPU:1*]
_classS
QOloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/Reshape_1*
_output_shapes
:*
T0
?
DMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/ShapeShape1My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/Relu"/device:GPU:1*
_output_shapes
:*
T0
?
FMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Shape_1Shape*My_GPU_1/FCU_muiltDense_x0/dropout/truediv"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
TMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/ShapeFMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
BMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/MulMulYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/tuple/control_dependency*My_GPU_1/FCU_muiltDense_x0/dropout/truediv"/device:GPU:1*
T0*
_output_shapes
:
?
BMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/SumSumBMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/MulTMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/BroadcastGradientArgs"/device:GPU:1*
_output_shapes
:*
T0
?
FMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/ReshapeReshapeBMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/SumDMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Shape"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
DMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Mul_1Mul1My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/ReluYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_1_grad/tuple/control_dependency"/device:GPU:1*
_output_shapes
:*
T0
?
DMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Sum_1SumDMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Mul_1VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
HMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Reshape_1ReshapeDMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Sum_1FMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Shape_1"/device:GPU:1*
T0*
_output_shapes
:
?
OMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/tuple/group_depsNoOpG^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/ReshapeI^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Reshape_1"/device:GPU:1
?
WMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/tuple/control_dependencyIdentityFMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/ReshapeP^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/tuple/group_deps"/device:GPU:1*Y
_classO
MKloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Reshape*
T0*'
_output_shapes
:????????? 
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/tuple/control_dependency_1IdentityHMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Reshape_1P^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
:*[
_classQ
OMloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/Reshape_1*
T0
?
RMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/Relu_grad/ReluGradReluGradWMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dropout/mul_grad/tuple/control_dependency1My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/Relu"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/ShapeShape<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1"/device:GPU:1*
_output_shapes
:*
T0
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Shape_1Shape:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub"/device:GPU:1*
T0*
_output_shapes
:
?
jMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Shape\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/SumSumRMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/Relu_grad/ReluGradjMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/BroadcastGradientArgs"/device:GPU:1*
_output_shapes
:*
T0
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/ReshapeReshapeXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/SumZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Shape"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Sum_1SumRMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/Relu_grad/ReluGradlMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Reshape_1ReshapeZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Sum_1\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Shape_1"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
eMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/tuple/group_depsNoOp]^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Reshape_^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Reshape_1"/device:GPU:1
?
mMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/tuple/control_dependencyIdentity\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Reshapef^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:????????? *o
_classe
caloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Reshape*
T0
?
oMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/tuple/control_dependency_1Identity^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Reshape_1f^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/tuple/group_deps"/device:GPU:1*q
_classg
ecloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/Reshape_1*'
_output_shapes
:????????? *
T0
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/ShapeShape(My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd"/device:GPU:1*
_output_shapes
:*
T0
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Shape_1Shape:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul"/device:GPU:1*
T0*
_output_shapes
:
?
jMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Shape\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/MulMulmMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/tuple/control_dependency:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/SumSumXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/MuljMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/ReshapeReshapeXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/SumZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Shape"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Mul_1Mul(My_GPU_1/FCU_muiltDense_x0/dense/BiasAddmMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/tuple/control_dependency"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Sum_1SumZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Mul_1lMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Reshape_1ReshapeZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Sum_1\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Shape_1"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
eMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/tuple/group_depsNoOp]^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Reshape_^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Reshape_1"/device:GPU:1
?
mMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/tuple/control_dependencyIdentity\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Reshapef^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:????????? *o
_classe
caloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Reshape*
T0
?
oMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/tuple/control_dependency_1Identity^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Reshape_1f^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/tuple/group_deps"/device:GPU:1*q
_classg
ecloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:????????? *
T0
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/ShapeShapeFCU_muiltDense_x0/beta/read"/device:GPU:1*
T0*
_output_shapes
:
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Shape_1Shape<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2"/device:GPU:1*
T0*
_output_shapes
:
?
hMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/ShapeZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/SumSumoMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/tuple/control_dependency_1hMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/ReshapeReshapeVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/SumXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Shape"/device:GPU:1*
T0*
_output_shapes
: 
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/NegNegoMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_1_grad/tuple/control_dependency_1"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Sum_1SumVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/NegjMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/BroadcastGradientArgs:1"/device:GPU:1*
T0*
_output_shapes
:
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Reshape_1ReshapeXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Sum_1ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Shape_1"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
cMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/tuple/group_depsNoOp[^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Reshape]^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Reshape_1"/device:GPU:1
?
kMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/tuple/control_dependencyIdentityZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Reshaped^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/tuple/group_deps"/device:GPU:1*m
_classc
a_loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Reshape*
T0*
_output_shapes
: 
?
mMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/tuple/control_dependency_1Identity\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Reshape_1d^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/tuple/group_deps"/device:GPU:1*o
_classe
caloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/Reshape_1*'
_output_shapes
:????????? *
T0
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/ShapeShape9My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean"/device:GPU:1*
_output_shapes
:*
T0
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Shape_1Shape:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul"/device:GPU:1*
T0*
_output_shapes
:
?
jMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Shape\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/MulMulmMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/tuple/control_dependency_1:My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/SumSumXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/MuljMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/BroadcastGradientArgs"/device:GPU:1*
_output_shapes
:*
T0
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/ReshapeReshapeXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/SumZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Shape"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Mul_1Mul9My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/meanmMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/tuple/control_dependency_1"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Sum_1SumZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Mul_1lMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Reshape_1ReshapeZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Sum_1\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Shape_1"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
eMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/tuple/group_depsNoOp]^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Reshape_^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Reshape_1"/device:GPU:1
?
mMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/tuple/control_dependencyIdentity\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Reshapef^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:?????????*
T0*o
_classe
caloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Reshape
?
oMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/tuple/control_dependency_1Identity^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Reshape_1f^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:????????? *q
_classg
ecloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/Reshape_1*
T0
?
My_GPU_1/gradients/AddN_3AddNoMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/tuple/control_dependency_1oMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/tuple/control_dependency_1"/device:GPU:1*
T0*
N*q
_classg
ecloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Reshape_1*'
_output_shapes
:????????? 
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/ShapeShape<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/Rsqrt"/device:GPU:1*
T0*
_output_shapes
:
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Shape_1ShapeFCU_muiltDense_x0/gamma/read"/device:GPU:1*
_output_shapes
:*
T0
?
hMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/ShapeZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/MulMulMy_GPU_1/gradients/AddN_3FCU_muiltDense_x0/gamma/read"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/SumSumVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/MulhMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/BroadcastGradientArgs"/device:GPU:1*
_output_shapes
:*
T0
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/ReshapeReshapeVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/SumXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Shape"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Mul_1Mul<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/RsqrtMy_GPU_1/gradients/AddN_3"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Sum_1SumXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Mul_1jMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Reshape_1ReshapeXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Sum_1ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Shape_1"/device:GPU:1*
_output_shapes
: *
T0
?
cMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/tuple/group_depsNoOp[^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Reshape]^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Reshape_1"/device:GPU:1
?
kMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/tuple/control_dependencyIdentityZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Reshaped^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:?????????*m
_classc
a_loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Reshape*
T0
?
mMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/tuple/control_dependency_1Identity\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Reshape_1d^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/tuple/group_deps"/device:GPU:1*o
_classe
caloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/Reshape_1*
T0*
_output_shapes
: 
?
^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/RsqrtkMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/tuple/control_dependency"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/ShapeShape=My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance"/device:GPU:1*
_output_shapes
:*
T0
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Shape_1Shape<My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add/y"/device:GPU:1*
T0*
_output_shapes
: 
?
hMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/ShapeZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/SumSum^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/Rsqrt_grad/RsqrtGradhMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/ReshapeReshapeVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/SumXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Shape"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Sum_1Sum^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/Rsqrt_grad/RsqrtGradjMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/BroadcastGradientArgs:1"/device:GPU:1*
T0*
_output_shapes
:
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Reshape_1ReshapeXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Sum_1ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Shape_1"/device:GPU:1*
_output_shapes
: *
T0
?
cMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/tuple/group_depsNoOp[^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Reshape]^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Reshape_1"/device:GPU:1
?
kMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/tuple/control_dependencyIdentityZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Reshaped^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/tuple/group_deps"/device:GPU:1*m
_classc
a_loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Reshape*'
_output_shapes
:?????????*
T0
?
mMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/tuple/control_dependency_1Identity\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Reshape_1d^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
: *
T0*o
_classe
caloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/Reshape_1
?
[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/ShapeShapeFMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference"/device:GPU:1*
T0*
_output_shapes
:
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/SizeConst"/device:GPU:1*
value	B :*
_output_shapes
: *n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape*
dtype0
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/addAddV2OMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance/reduction_indicesZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Size"/device:GPU:1*
T0*
_output_shapes
:*n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/modFloorModYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/addZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Size"/device:GPU:1*n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape*
_output_shapes
:*
T0
?
]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape_1Const"/device:GPU:1*
_output_shapes
:*
dtype0*n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape*
valueB:
?
aMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/range/startConst"/device:GPU:1*
_output_shapes
: *
value	B : *
dtype0*n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape
?
aMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/range/deltaConst"/device:GPU:1*
_output_shapes
: *n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape*
value	B :*
dtype0
?
[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/rangeRangeaMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/range/startZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/SizeaMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/range/delta"/device:GPU:1*
_output_shapes
:*n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape
?
`My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Fill/valueConst"/device:GPU:1*
value	B :*
dtype0*n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape*
_output_shapes
: 
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/FillFill]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape_1`My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Fill/value"/device:GPU:1*n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape*
_output_shapes
:*
T0
?
cMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/DynamicStitchDynamicStitch[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/rangeYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/mod[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/ShapeZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Fill"/device:GPU:1*
T0*n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape*
N*
_output_shapes
:
?
_My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Maximum/yConst"/device:GPU:1*
_output_shapes
: *
dtype0*
value	B :*n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape
?
]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/MaximumMaximumcMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/DynamicStitch_My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Maximum/y"/device:GPU:1*
T0*n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape*
_output_shapes
:
?
^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/floordivFloorDiv[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Maximum"/device:GPU:1*
_output_shapes
:*
T0*n
_classd
b`loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape
?
]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/ReshapeReshapekMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/add_grad/tuple/control_dependencycMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/DynamicStitch"/device:GPU:1*0
_output_shapes
:??????????????????*
T0
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/TileTile]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Reshape^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/floordiv"/device:GPU:1*0
_output_shapes
:??????????????????*
T0
?
]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape_2ShapeFMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference"/device:GPU:1*
T0*
_output_shapes
:
?
]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape_3Shape=My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance"/device:GPU:1*
T0*
_output_shapes
:
?
[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/ConstConst"/device:GPU:1*
dtype0*
valueB: *
_output_shapes
:
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/ProdProd]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape_2[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Const"/device:GPU:1*
T0*
_output_shapes
: 
?
]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Const_1Const"/device:GPU:1*
dtype0*
valueB: *
_output_shapes
:
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Prod_1Prod]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Shape_3]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Const_1"/device:GPU:1*
_output_shapes
: *
T0
?
aMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Maximum_1/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
value	B :
?
_My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Maximum_1Maximum\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Prod_1aMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Maximum_1/y"/device:GPU:1*
_output_shapes
: *
T0
?
`My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/floordiv_1FloorDivZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Prod_My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Maximum_1"/device:GPU:1*
_output_shapes
: *
T0
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/CastCast`My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/floordiv_1"/device:GPU:1*
_output_shapes
: *

DstT0*

SrcT0
?
]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/truedivRealDivZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/TileZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/Cast"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
eMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/scalarConst^^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/truediv"/device:GPU:1*
dtype0*
valueB
 *   @*
_output_shapes
: 
?
bMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/MulMuleMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/scalar]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/truediv"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
bMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/subSub(My_GPU_1/FCU_muiltDense_x0/dense/BiasAddAMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/StopGradient^^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/variance_grad/truediv"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
dMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/mul_1MulbMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/MulbMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/sub"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
dMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/ShapeShape(My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd"/device:GPU:1*
_output_shapes
:*
T0
?
fMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Shape_1ShapeAMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/StopGradient"/device:GPU:1*
_output_shapes
:*
T0
?
tMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsdMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/ShapefMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
bMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/SumSumdMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/mul_1tMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
fMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/ReshapeReshapebMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/SumdMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Shape"/device:GPU:1*'
_output_shapes
:????????? *
T0
?
dMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Sum_1SumdMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/mul_1vMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
hMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Reshape_1ReshapedMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Sum_1fMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Shape_1"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
bMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/NegNeghMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Reshape_1"/device:GPU:1*
T0*'
_output_shapes
:?????????
?
oMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/tuple/group_depsNoOpc^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Negg^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Reshape"/device:GPU:1
?
wMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/tuple/control_dependencyIdentityfMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Reshapep^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/tuple/group_deps"/device:GPU:1*y
_classo
mkloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Reshape*'
_output_shapes
:????????? *
T0
?
yMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/tuple/control_dependency_1IdentitybMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Negp^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/tuple/group_deps"/device:GPU:1*
T0*'
_output_shapes
:?????????*u
_classk
igloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/Neg
?
WMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/ShapeShape(My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd"/device:GPU:1*
T0*
_output_shapes
:
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/SizeConst"/device:GPU:1*
dtype0*
value	B :*
_output_shapes
: *j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape
?
UMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/addAddV2KMy_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean/reduction_indicesVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Size"/device:GPU:1*j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape*
T0*
_output_shapes
:
?
UMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/modFloorModUMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/addVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Size"/device:GPU:1*
T0*j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape*
_output_shapes
:
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape_1Const"/device:GPU:1*
_output_shapes
:*
dtype0*
valueB:*j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape
?
]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/range/startConst"/device:GPU:1*
dtype0*j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape*
_output_shapes
: *
value	B : 
?
]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/range/deltaConst"/device:GPU:1*
dtype0*
_output_shapes
: *j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape*
value	B :
?
WMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/rangeRange]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/range/startVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Size]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/range/delta"/device:GPU:1*
_output_shapes
:*j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Fill/valueConst"/device:GPU:1*
_output_shapes
: *j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape*
value	B :*
dtype0
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/FillFillYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape_1\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Fill/value"/device:GPU:1*
_output_shapes
:*j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape*
T0
?
_My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/DynamicStitchDynamicStitchWMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/rangeUMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/modWMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/ShapeVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Fill"/device:GPU:1*
_output_shapes
:*j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape*
N*
T0
?
[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Maximum/yConst"/device:GPU:1*
value	B :*j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/MaximumMaximum_My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/DynamicStitch[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Maximum/y"/device:GPU:1*
T0*
_output_shapes
:*j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/floordivFloorDivWMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/ShapeYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Maximum"/device:GPU:1*
T0*j
_class`
^\loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape*
_output_shapes
:
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/ReshapeReshapemMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_2_grad/tuple/control_dependency_My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/DynamicStitch"/device:GPU:1*0
_output_shapes
:??????????????????*
T0
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/TileTileYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/ReshapeZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/floordiv"/device:GPU:1*
T0*0
_output_shapes
:??????????????????
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape_2Shape(My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd"/device:GPU:1*
_output_shapes
:*
T0
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape_3Shape9My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean"/device:GPU:1*
T0*
_output_shapes
:
?
WMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/ConstConst"/device:GPU:1*
dtype0*
_output_shapes
:*
valueB: 
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/ProdProdYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape_2WMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Const"/device:GPU:1*
_output_shapes
: *
T0
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Const_1Const"/device:GPU:1*
_output_shapes
:*
valueB: *
dtype0
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Prod_1ProdYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Shape_3YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Const_1"/device:GPU:1*
T0*
_output_shapes
: 
?
]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Maximum_1/yConst"/device:GPU:1*
_output_shapes
: *
value	B :*
dtype0
?
[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Maximum_1MaximumXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Prod_1]My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Maximum_1/y"/device:GPU:1*
_output_shapes
: *
T0
?
\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/floordiv_1FloorDivVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Prod[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Maximum_1"/device:GPU:1*
_output_shapes
: *
T0
?
VMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/CastCast\My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/floordiv_1"/device:GPU:1*
_output_shapes
: *

SrcT0*

DstT0
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/truedivRealDivVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/TileVMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/Cast"/device:GPU:1*
T0*'
_output_shapes
:????????? 
?
My_GPU_1/gradients/AddN_4AddNmMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/tuple/control_dependencywMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/SquaredDifference_grad/tuple/control_dependencyYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/moments/mean_grad/truediv"/device:GPU:1*
N*o
_classe
caloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Reshape*
T0*'
_output_shapes
:????????? 
?
LMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/BiasAddGradBiasAddGradMy_GPU_1/gradients/AddN_4"/device:GPU:1*
_output_shapes
: *
T0
?
QMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/tuple/group_depsNoOp^My_GPU_1/gradients/AddN_4M^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/BiasAddGrad"/device:GPU:1
?
YMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/tuple/control_dependencyIdentityMy_GPU_1/gradients/AddN_4R^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/tuple/group_deps"/device:GPU:1*
T0*'
_output_shapes
:????????? *o
_classe
caloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_1_grad/Reshape
?
[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/tuple/control_dependency_1IdentityLMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/BiasAddGradR^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/tuple/group_deps"/device:GPU:1*_
_classU
SQloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
?
FMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/MatMulMatMulYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/tuple/control_dependency#FCU_muiltDense_x0/dense/kernel/read"/device:GPU:1*'
_output_shapes
:????????? *
T0*
transpose_b(
?
HMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/MatMul_1MatMulMy_GPU_1/dense/BiasAddYMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/tuple/control_dependency"/device:GPU:1*
_output_shapes

:  *
T0*
transpose_a(
?
PMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/tuple/group_depsNoOpG^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/MatMulI^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/MatMul_1"/device:GPU:1
?
XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/tuple/control_dependencyIdentityFMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/MatMulQ^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/tuple/group_deps"/device:GPU:1*
T0*'
_output_shapes
:????????? *Y
_classO
MKloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/MatMul
?
ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/tuple/control_dependency_1IdentityHMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/MatMul_1Q^My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/tuple/group_deps"/device:GPU:1*[
_classQ
OMloc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/MatMul_1*
_output_shapes

:  *
T0
?
My_GPU_1/gradients/AddN_5AddN?My_GPU_1/gradients/My_GPU_1/add_grad/tuple/control_dependency_1XMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/tuple/control_dependency"/device:GPU:1*'
_output_shapes
:????????? *
N*
T0*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_grad/Reshape_1
?
:My_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/BiasAddGradBiasAddGradMy_GPU_1/gradients/AddN_5"/device:GPU:1*
T0*
_output_shapes
: 
?
?My_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/tuple/group_depsNoOp^My_GPU_1/gradients/AddN_5;^My_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/BiasAddGrad"/device:GPU:1
?
GMy_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/tuple/control_dependencyIdentityMy_GPU_1/gradients/AddN_5@^My_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/tuple/group_deps"/device:GPU:1*A
_class7
53loc:@My_GPU_1/gradients/My_GPU_1/add_grad/Reshape_1*
T0*'
_output_shapes
:????????? 
?
IMy_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/tuple/control_dependency_1Identity:My_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/BiasAddGrad@^My_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/tuple/group_deps"/device:GPU:1*
T0*M
_classC
A?loc:@My_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
?
My_GPU_1/gradients/AddN_6AddNXMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Square_grad/Mul_1ZMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/MatMul_grad/tuple/control_dependency_1"/device:GPU:1*
N*
_output_shapes

:  *k
_classa
_]loc:@My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/Square_grad/Mul_1*
T0
?
4My_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/MatMulMatMulGMy_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read"/device:GPU:1*
T0*(
_output_shapes
:??????????*
transpose_b(
?
6My_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/MatMul_1MatMul!My_GPU_1/Conv_out__/dropout/mul_1GMy_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/tuple/control_dependency"/device:GPU:1*
T0*
transpose_a(*
_output_shapes
:	? 
?
>My_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/tuple/group_depsNoOp5^My_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/MatMul7^My_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/MatMul_1"/device:GPU:1
?
FMy_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/tuple/control_dependencyIdentity4My_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/MatMul?^My_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/tuple/group_deps"/device:GPU:1*
T0*G
_class=
;9loc:@My_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
HMy_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/tuple/control_dependency_1Identity6My_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/MatMul_1?^My_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/tuple/group_deps"/device:GPU:1*
T0*I
_class?
=;loc:@My_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/MatMul_1*
_output_shapes
:	? 
?
?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/ShapeShapeMy_GPU_1/Conv_out__/dropout/mul"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
AMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Shape_1Shape My_GPU_1/Conv_out__/dropout/Cast"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
OMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/ShapeAMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
=My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/MulMulFMy_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/tuple/control_dependency My_GPU_1/Conv_out__/dropout/Cast"/device:GPU:1*
_output_shapes
:*
T0
?
=My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/SumSum=My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/MulOMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/BroadcastGradientArgs"/device:GPU:1*
_output_shapes
:*
T0
?
AMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/ReshapeReshape=My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Sum?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Shape"/device:GPU:1*
T0*
_output_shapes
:
?
?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Mul_1MulMy_GPU_1/Conv_out__/dropout/mulFMy_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/tuple/control_dependency"/device:GPU:1*
_output_shapes
:*
T0
?
?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Sum_1Sum?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Mul_1QMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:1*
T0*
_output_shapes
:
?
CMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Reshape_1Reshape?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Sum_1AMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Shape_1"/device:GPU:1*
_output_shapes
:*
T0
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/tuple/group_depsNoOpB^My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/ReshapeD^My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Reshape_1"/device:GPU:1
?
RMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/tuple/control_dependencyIdentityAMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/ReshapeK^My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/tuple/group_deps"/device:GPU:1*
T0*T
_classJ
HFloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Reshape*
_output_shapes
:
?
TMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/tuple/control_dependency_1IdentityCMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Reshape_1K^My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/tuple/group_deps"/device:GPU:1*
T0*
_output_shapes
:*V
_classL
JHloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/Reshape_1
?
My_GPU_1/gradients/AddN_7AddNFMy_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Square_grad/Mul_1HMy_GPU_1/gradients/My_GPU_1/dense/MatMul_grad/tuple/control_dependency_1"/device:GPU:1*
T0*
N*Y
_classO
MKloc:@My_GPU_1/gradients/My_GPU_1/dense/kernel/Regularizer/Square_grad/Mul_1*
_output_shapes
:	? 
?
=My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/ShapeShape#My_GPU_1/Conv_out__/Conv_out__/Relu"/device:GPU:1*
T0*
_output_shapes
:
?
?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Shape_1Shape#My_GPU_1/Conv_out__/dropout/truediv"/device:GPU:1*
T0*#
_output_shapes
:?????????
?
MMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Shape?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
;My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/MulMulRMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/tuple/control_dependency#My_GPU_1/Conv_out__/dropout/truediv"/device:GPU:1*
T0*
_output_shapes
:
?
;My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/SumSum;My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/MulMMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/ReshapeReshape;My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Sum=My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Shape"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
=My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Mul_1Mul#My_GPU_1/Conv_out__/Conv_out__/ReluRMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_1_grad/tuple/control_dependency"/device:GPU:1*
T0*
_output_shapes
:
?
=My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Sum_1Sum=My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Mul_1OMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/BroadcastGradientArgs:1"/device:GPU:1*
T0*
_output_shapes
:
?
AMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Reshape_1Reshape=My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Sum_1?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Shape_1"/device:GPU:1*
T0*
_output_shapes
:
?
HMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/tuple/group_depsNoOp@^My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/ReshapeB^My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Reshape_1"/device:GPU:1
?
PMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/tuple/control_dependencyIdentity?My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/ReshapeI^My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/tuple/group_deps"/device:GPU:1*
T0*R
_classH
FDloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Reshape*(
_output_shapes
:??????????
?
RMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/tuple/control_dependency_1IdentityAMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Reshape_1I^My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes
:*T
_classJ
HFloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/Reshape_1*
T0
?
My_GPU_1/gradients/AddN_8AddN\My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/tuple/control_dependencyPMy_GPU_1/gradients/My_GPU_1/Conv_out__/dropout/mul_grad/tuple/control_dependency"/device:GPU:1*
N*
T0*(
_output_shapes
:??????????*]
_classS
QOloc:@My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/MatMul_grad/MatMul
?
DMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/Relu_grad/ReluGradReluGradMy_GPU_1/gradients/AddN_8#My_GPU_1/Conv_out__/Conv_out__/Relu"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/ShapeShape.My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1"/device:GPU:1*
T0*
_output_shapes
:
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/Shape_1Shape,My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub"/device:GPU:1*
T0*
_output_shapes
:
?
\My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/ShapeNMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/SumSumDMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/Relu_grad/ReluGrad\My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/BroadcastGradientArgs"/device:GPU:1*
_output_shapes
:*
T0
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/ReshapeReshapeJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/SumLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/Shape"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/Sum_1SumDMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/Relu_grad/ReluGrad^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
PMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/Reshape_1ReshapeLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/Sum_1NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/Shape_1"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
WMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/tuple/group_depsNoOpO^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/ReshapeQ^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/Reshape_1"/device:GPU:1
?
_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/tuple/control_dependencyIdentityNMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/ReshapeX^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/tuple/group_deps"/device:GPU:1*
T0*a
_classW
USloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/Reshape*(
_output_shapes
:??????????
?
aMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/tuple/control_dependency_1IdentityPMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/Reshape_1X^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/tuple/group_deps"/device:GPU:1*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/Reshape_1*
T0*(
_output_shapes
:??????????
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/ShapeShapeMy_GPU_1/concat/concat"/device:GPU:1*
T0*
_output_shapes
:
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Shape_1Shape,My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul"/device:GPU:1*
_output_shapes
:*
T0
?
\My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/ShapeNMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/MulMul_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/tuple/control_dependency,My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/SumSumJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Mul\My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/ReshapeReshapeJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/SumLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Shape"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Mul_1MulMy_GPU_1/concat/concat_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/tuple/control_dependency"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Sum_1SumLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Mul_1^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
PMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Reshape_1ReshapeLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Sum_1NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Shape_1"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
WMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/tuple/group_depsNoOpO^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/ReshapeQ^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Reshape_1"/device:GPU:1
?
_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/tuple/control_dependencyIdentityNMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/ReshapeX^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/tuple/group_deps"/device:GPU:1*(
_output_shapes
:??????????*
T0*a
_classW
USloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Reshape
?
aMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityPMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Reshape_1X^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/tuple/group_deps"/device:GPU:1*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Reshape_1*(
_output_shapes
:??????????*
T0
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/ShapeShapeConv_out__/beta/read"/device:GPU:1*
_output_shapes
:*
T0
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Shape_1Shape.My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2"/device:GPU:1*
_output_shapes
:*
T0
?
ZMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgsJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/ShapeLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
HMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/SumSumaMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/tuple/control_dependency_1ZMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/BroadcastGradientArgs"/device:GPU:1*
_output_shapes
:*
T0
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/ReshapeReshapeHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/SumJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Shape"/device:GPU:1*
_output_shapes	
:?*
T0
?
HMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/NegNegaMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_1_grad/tuple/control_dependency_1"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Sum_1SumHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Neg\My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Reshape_1ReshapeJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Sum_1LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Shape_1"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
UMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/tuple/group_depsNoOpM^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/ReshapeO^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Reshape_1"/device:GPU:1
?
]My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/tuple/control_dependencyIdentityLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/ReshapeV^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/tuple/group_deps"/device:GPU:1*
_output_shapes	
:?*
T0*_
_classU
SQloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Reshape
?
_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/tuple/control_dependency_1IdentityNMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Reshape_1V^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/tuple/group_deps"/device:GPU:1*
T0*(
_output_shapes
:??????????*a
_classW
USloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/Reshape_1
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/ShapeShape+My_GPU_1/Conv_out__/Conv_out__/moments/mean"/device:GPU:1*
T0*
_output_shapes
:
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Shape_1Shape,My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul"/device:GPU:1*
_output_shapes
:*
T0
?
\My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/ShapeNMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/MulMul_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/tuple/control_dependency_1,My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/SumSumJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Mul\My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/BroadcastGradientArgs"/device:GPU:1*
_output_shapes
:*
T0
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/ReshapeReshapeJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/SumLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Shape"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Mul_1Mul+My_GPU_1/Conv_out__/Conv_out__/moments/mean_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/tuple/control_dependency_1"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Sum_1SumLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Mul_1^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
PMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Reshape_1ReshapeLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Sum_1NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Shape_1"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
WMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/tuple/group_depsNoOpO^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/ReshapeQ^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Reshape_1"/device:GPU:1
?
_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/tuple/control_dependencyIdentityNMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/ReshapeX^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:?????????*
T0*a
_classW
USloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Reshape
?
aMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityPMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Reshape_1X^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/tuple/group_deps"/device:GPU:1*
T0*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/Reshape_1*(
_output_shapes
:??????????
?
My_GPU_1/gradients/AddN_9AddNaMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/tuple/control_dependency_1aMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/tuple/control_dependency_1"/device:GPU:1*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Reshape_1*
T0*(
_output_shapes
:??????????*
N
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/ShapeShape.My_GPU_1/Conv_out__/Conv_out__/batchnorm/Rsqrt"/device:GPU:1*
T0*
_output_shapes
:
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Shape_1ShapeConv_out__/gamma/read"/device:GPU:1*
_output_shapes
:*
T0
?
ZMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/ShapeLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
HMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/MulMulMy_GPU_1/gradients/AddN_9Conv_out__/gamma/read"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
HMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/SumSumHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/MulZMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/BroadcastGradientArgs"/device:GPU:1*
_output_shapes
:*
T0
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/ReshapeReshapeHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/SumJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Shape"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Mul_1Mul.My_GPU_1/Conv_out__/Conv_out__/batchnorm/RsqrtMy_GPU_1/gradients/AddN_9"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Sum_1SumJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Mul_1\My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Reshape_1ReshapeJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Sum_1LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Shape_1"/device:GPU:1*
T0*
_output_shapes	
:?
?
UMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/tuple/group_depsNoOpM^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/ReshapeO^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Reshape_1"/device:GPU:1
?
]My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/tuple/control_dependencyIdentityLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/ReshapeV^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:?????????*
T0*_
_classU
SQloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Reshape
?
_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/tuple/control_dependency_1IdentityNMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Reshape_1V^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/tuple/group_deps"/device:GPU:1*
_output_shapes	
:?*a
_classW
USloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/Reshape_1*
T0
?
PMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad.My_GPU_1/Conv_out__/Conv_out__/batchnorm/Rsqrt]My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/tuple/control_dependency"/device:GPU:1*
T0*'
_output_shapes
:?????????
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/ShapeShape/My_GPU_1/Conv_out__/Conv_out__/moments/variance"/device:GPU:1*
T0*
_output_shapes
:
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/Shape_1Shape.My_GPU_1/Conv_out__/Conv_out__/batchnorm/add/y"/device:GPU:1*
T0*
_output_shapes
: 
?
ZMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgsJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/ShapeLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
HMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/SumSumPMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/Rsqrt_grad/RsqrtGradZMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/ReshapeReshapeHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/SumJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/Shape"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/Sum_1SumPMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/Rsqrt_grad/RsqrtGrad\My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/Reshape_1ReshapeJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/Sum_1LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/Shape_1"/device:GPU:1*
T0*
_output_shapes
: 
?
UMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/tuple/group_depsNoOpM^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/ReshapeO^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/Reshape_1"/device:GPU:1
?
]My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/tuple/control_dependencyIdentityLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/ReshapeV^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:?????????*_
_classU
SQloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/Reshape*
T0
?
_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/tuple/control_dependency_1IdentityNMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/Reshape_1V^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/tuple/group_deps"/device:GPU:1*a
_classW
USloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/Reshape_1*
_output_shapes
: *
T0
?
MMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/ShapeShape8My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference"/device:GPU:1*
_output_shapes
:*
T0
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/SizeConst"/device:GPU:1*
value	B :*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
_output_shapes
: *
dtype0
?
KMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/addAddV2AMy_GPU_1/Conv_out__/Conv_out__/moments/variance/reduction_indicesLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Size"/device:GPU:1*
T0*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
_output_shapes
:
?
KMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/modFloorModKMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/addLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Size"/device:GPU:1*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
_output_shapes
:*
T0
?
OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape_1Const"/device:GPU:1*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
valueB:*
_output_shapes
:*
dtype0
?
SMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/range/startConst"/device:GPU:1*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
_output_shapes
: *
value	B : *
dtype0
?
SMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/range/deltaConst"/device:GPU:1*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
dtype0*
value	B :*
_output_shapes
: 
?
MMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/rangeRangeSMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/range/startLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/SizeSMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/range/delta"/device:GPU:1*
_output_shapes
:*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape
?
RMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Fill/valueConst"/device:GPU:1*
dtype0*
_output_shapes
: *`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
value	B :
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/FillFillOMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape_1RMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Fill/value"/device:GPU:1*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
_output_shapes
:*
T0
?
UMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/DynamicStitchDynamicStitchMMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/rangeKMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/modMMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/ShapeLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Fill"/device:GPU:1*
T0*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
N*
_output_shapes
:
?
QMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Maximum/yConst"/device:GPU:1*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/MaximumMaximumUMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/DynamicStitchQMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Maximum/y"/device:GPU:1*
_output_shapes
:*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
T0
?
PMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/floordivFloorDivMMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/ShapeOMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Maximum"/device:GPU:1*
_output_shapes
:*`
_classV
TRloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape*
T0
?
OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/ReshapeReshape]My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/add_grad/tuple/control_dependencyUMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/DynamicStitch"/device:GPU:1*
T0*0
_output_shapes
:??????????????????
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/TileTileOMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/ReshapePMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/floordiv"/device:GPU:1*0
_output_shapes
:??????????????????*
T0
?
OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape_2Shape8My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference"/device:GPU:1*
T0*
_output_shapes
:
?
OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape_3Shape/My_GPU_1/Conv_out__/Conv_out__/moments/variance"/device:GPU:1*
T0*
_output_shapes
:
?
MMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/ConstConst"/device:GPU:1*
_output_shapes
:*
valueB: *
dtype0
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/ProdProdOMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape_2MMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Const"/device:GPU:1*
T0*
_output_shapes
: 
?
OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Const_1Const"/device:GPU:1*
valueB: *
dtype0*
_output_shapes
:
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Prod_1ProdOMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Shape_3OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Const_1"/device:GPU:1*
_output_shapes
: *
T0
?
SMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Maximum_1/yConst"/device:GPU:1*
dtype0*
value	B :*
_output_shapes
: 
?
QMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Maximum_1MaximumNMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Prod_1SMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Maximum_1/y"/device:GPU:1*
T0*
_output_shapes
: 
?
RMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/floordiv_1FloorDivLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/ProdQMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Maximum_1"/device:GPU:1*
_output_shapes
: *
T0
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/CastCastRMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/floordiv_1"/device:GPU:1*

DstT0*
_output_shapes
: *

SrcT0
?
OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/truedivRealDivLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/TileLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/Cast"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
WMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/scalarConstP^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/truediv"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB
 *   @
?
TMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/MulMulWMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/scalarOMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/truediv"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
TMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/subSubMy_GPU_1/concat/concat3My_GPU_1/Conv_out__/Conv_out__/moments/StopGradientP^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/variance_grad/truediv"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
VMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/mul_1MulTMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/MulTMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/sub"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
VMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/ShapeShapeMy_GPU_1/concat/concat"/device:GPU:1*
_output_shapes
:*
T0
?
XMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Shape_1Shape3My_GPU_1/Conv_out__/Conv_out__/moments/StopGradient"/device:GPU:1*
_output_shapes
:*
T0
?
fMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsVMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/ShapeXMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Shape_1"/device:GPU:1*2
_output_shapes 
:?????????:?????????
?
TMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/SumSumVMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/mul_1fMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/BroadcastGradientArgs"/device:GPU:1*
T0*
_output_shapes
:
?
XMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/ReshapeReshapeTMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/SumVMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Shape"/device:GPU:1*(
_output_shapes
:??????????*
T0
?
VMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Sum_1SumVMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/mul_1hMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/BroadcastGradientArgs:1"/device:GPU:1*
_output_shapes
:*
T0
?
ZMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Reshape_1ReshapeVMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Sum_1XMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Shape_1"/device:GPU:1*'
_output_shapes
:?????????*
T0
?
TMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/NegNegZMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Reshape_1"/device:GPU:1*
T0*'
_output_shapes
:?????????
?
aMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/tuple/group_depsNoOpU^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/NegY^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Reshape"/device:GPU:1
?
iMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/tuple/control_dependencyIdentityXMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Reshapeb^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/tuple/group_deps"/device:GPU:1*
T0*k
_classa
_]loc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Reshape*(
_output_shapes
:??????????
?
kMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityTMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Negb^My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:?????????*
T0*g
_class]
[Yloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/Neg
?
IMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/ShapeShapeMy_GPU_1/concat/concat"/device:GPU:1*
_output_shapes
:*
T0
?
HMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/SizeConst"/device:GPU:1*
value	B :*
_output_shapes
: *\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape*
dtype0
?
GMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/addAddV2=My_GPU_1/Conv_out__/Conv_out__/moments/mean/reduction_indicesHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Size"/device:GPU:1*
T0*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape*
_output_shapes
:
?
GMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/modFloorModGMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/addHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Size"/device:GPU:1*
_output_shapes
:*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape*
T0
?
KMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape_1Const"/device:GPU:1*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape*
dtype0*
valueB:*
_output_shapes
:
?
OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/range/startConst"/device:GPU:1*
_output_shapes
: *\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape*
dtype0*
value	B : 
?
OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/range/deltaConst"/device:GPU:1*
_output_shapes
: *\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape*
value	B :*
dtype0
?
IMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/rangeRangeOMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/range/startHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/SizeOMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/range/delta"/device:GPU:1*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape*
_output_shapes
:
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Fill/valueConst"/device:GPU:1*
_output_shapes
: *
dtype0*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape*
value	B :
?
HMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/FillFillKMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape_1NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Fill/value"/device:GPU:1*
T0*
_output_shapes
:*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape
?
QMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/DynamicStitchDynamicStitchIMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/rangeGMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/modIMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/ShapeHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Fill"/device:GPU:1*
_output_shapes
:*
N*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape*
T0
?
MMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Maximum/yConst"/device:GPU:1*
value	B :*
_output_shapes
: *
dtype0*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape
?
KMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/MaximumMaximumQMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/DynamicStitchMMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Maximum/y"/device:GPU:1*
T0*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape*
_output_shapes
:
?
LMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/floordivFloorDivIMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/ShapeKMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Maximum"/device:GPU:1*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape*
T0*
_output_shapes
:
?
KMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/ReshapeReshape_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_2_grad/tuple/control_dependencyQMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/DynamicStitch"/device:GPU:1*0
_output_shapes
:??????????????????*
T0
?
HMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/TileTileKMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/ReshapeLMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/floordiv"/device:GPU:1*0
_output_shapes
:??????????????????*
T0
?
KMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape_2ShapeMy_GPU_1/concat/concat"/device:GPU:1*
T0*
_output_shapes
:
?
KMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape_3Shape+My_GPU_1/Conv_out__/Conv_out__/moments/mean"/device:GPU:1*
T0*
_output_shapes
:
?
IMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/ConstConst"/device:GPU:1*
dtype0*
valueB: *
_output_shapes
:
?
HMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/ProdProdKMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape_2IMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Const"/device:GPU:1*
_output_shapes
: *
T0
?
KMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Const_1Const"/device:GPU:1*
dtype0*
valueB: *
_output_shapes
:
?
JMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Prod_1ProdKMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Shape_3KMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Const_1"/device:GPU:1*
T0*
_output_shapes
: 
?
OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Maximum_1/yConst"/device:GPU:1*
value	B :*
dtype0*
_output_shapes
: 
?
MMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Maximum_1MaximumJMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Prod_1OMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Maximum_1/y"/device:GPU:1*
_output_shapes
: *
T0
?
NMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/floordiv_1FloorDivHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/ProdMMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Maximum_1"/device:GPU:1*
_output_shapes
: *
T0
?
HMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/CastCastNMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/floordiv_1"/device:GPU:1*

DstT0*

SrcT0*
_output_shapes
: 
?
KMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/truedivRealDivHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/TileHMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/Cast"/device:GPU:1*
T0*(
_output_shapes
:??????????
?
My_GPU_1/gradients/AddN_10AddN_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/tuple/control_dependencyiMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/SquaredDifference_grad/tuple/control_dependencyKMy_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/moments/mean_grad/truediv"/device:GPU:1*
T0*(
_output_shapes
:??????????*a
_classW
USloc:@My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_1_grad/Reshape*
N
?
;My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/Reshape_grad/ShapeShape"My_GPU_1/CCN_1Conv_x0/convB21/Relu"/device:GPU:1*
_output_shapes
:*
T0
?
=My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/Reshape_grad/ReshapeReshapeMy_GPU_1/gradients/AddN_10;My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/Reshape_grad/Shape"/device:GPU:1*,
_output_shapes
:??????????*
T0
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/Relu_grad/ReluGradReluGrad=My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/Reshape_grad/Reshape"My_GPU_1/CCN_1Conv_x0/convB21/Relu"/device:GPU:1*,
_output_shapes
:??????????*
T0
?
IMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd_grad/BiasAddGradBiasAddGradCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/Relu_grad/ReluGrad"/device:GPU:1*
_output_shapes	
:?*
T0
?
NMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd_grad/tuple/group_depsNoOpJ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd_grad/BiasAddGradD^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/Relu_grad/ReluGrad"/device:GPU:1
?
VMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd_grad/tuple/control_dependencyIdentityCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/Relu_grad/ReluGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd_grad/tuple/group_deps"/device:GPU:1*,
_output_shapes
:??????????*
T0*V
_classL
JHloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/Relu_grad/ReluGrad
?
XMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd_grad/tuple/control_dependency_1IdentityIMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd_grad/BiasAddGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd_grad/tuple/group_deps"/device:GPU:1*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?*
T0
?
JMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/Squeeze_grad/ShapeShape$My_GPU_1/CCN_1Conv_x0/convB21/conv1d"/device:GPU:1*
T0*
_output_shapes
:
?
LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/Squeeze_grad/ReshapeReshapeVMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd_grad/tuple/control_dependencyJMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/Squeeze_grad/Shape"/device:GPU:1*
T0*0
_output_shapes
:??????????
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/ShapeNShapeN/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_1"/device:GPU:1*
T0* 
_output_shapes
::*
N
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/Conv2DBackpropInputConv2DBackpropInputCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/ShapeN1My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/Squeeze_grad/Reshape"/device:GPU:1*
strides
*0
_output_shapes
:??????????*
T0*
paddingSAME
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/Conv2DBackpropFilterConv2DBackpropFilter/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDimsEMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/ShapeN:1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/Squeeze_grad/Reshape"/device:GPU:1*
T0*
strides
*
paddingSAME*(
_output_shapes
:??
?
MMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/tuple/group_depsNoOpR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/Conv2DBackpropFilterQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/Conv2DBackpropInput"/device:GPU:1
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/Conv2DBackpropInputN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/tuple/group_deps"/device:GPU:1*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/Conv2DBackpropInput*0
_output_shapes
:??????????*
T0
?
WMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/tuple/control_dependency_1IdentityQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/Conv2DBackpropFilterN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/tuple/group_deps"/device:GPU:1*
T0*(
_output_shapes
:??*d
_classZ
XVloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/Conv2DBackpropFilter
?
MMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_grad/ShapeShape%My_GPU_1/CCN_1Conv_x0/poolB11/Squeeze"/device:GPU:1*
_output_shapes
:*
T0
?
OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_grad/ReshapeReshapeUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/tuple/control_dependencyMMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_grad/Shape"/device:GPU:1*,
_output_shapes
:??????????*
T0
?
OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_1_grad/ShapeConst"/device:GPU:1*
dtype0*!
valueB"         *
_output_shapes
:
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_1_grad/ReshapeReshapeWMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d_grad/tuple/control_dependency_1OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_1_grad/Shape"/device:GPU:1*$
_output_shapes
:??*
T0
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB11/Squeeze_grad/ShapeShapeMy_GPU_1/CCN_1Conv_x0/poolB11"/device:GPU:1*
T0*
_output_shapes
:
?
EMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB11/Squeeze_grad/ReshapeReshapeOMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_grad/ReshapeCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB11/Squeeze_grad/Shape"/device:GPU:1*0
_output_shapes
:??????????*
T0
?
My_GPU_1/gradients/AddN_11AddNUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Square_grad/Mul_1QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/conv1d/ExpandDims_1_grad/Reshape"/device:GPU:1*
N*
T0*h
_class^
\Zloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/Square_grad/Mul_1*$
_output_shapes
:??
?
AMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB11_grad/MaxPoolGradMaxPoolGrad(My_GPU_1/CCN_1Conv_x0/poolB11/ExpandDimsMy_GPU_1/CCN_1Conv_x0/poolB11EMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB11/Squeeze_grad/Reshape"/device:GPU:1*
strides
*
ksize
*
paddingSAME*0
_output_shapes
:??????????
?
FMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB11/ExpandDims_grad/ShapeShape"My_GPU_1/CCN_1Conv_x0/convB11/Relu"/device:GPU:1*
_output_shapes
:*
T0
?
HMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB11/ExpandDims_grad/ReshapeReshapeAMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB11_grad/MaxPoolGradFMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB11/ExpandDims_grad/Shape"/device:GPU:1*,
_output_shapes
:??????????*
T0
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/Relu_grad/ReluGradReluGradHMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB11/ExpandDims_grad/Reshape"My_GPU_1/CCN_1Conv_x0/convB11/Relu"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
IMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd_grad/BiasAddGradBiasAddGradCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/Relu_grad/ReluGrad"/device:GPU:1*
_output_shapes	
:?*
T0
?
NMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd_grad/tuple/group_depsNoOpJ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd_grad/BiasAddGradD^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/Relu_grad/ReluGrad"/device:GPU:1
?
VMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd_grad/tuple/control_dependencyIdentityCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/Relu_grad/ReluGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd_grad/tuple/group_deps"/device:GPU:1*,
_output_shapes
:??????????*
T0*V
_classL
JHloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/Relu_grad/ReluGrad
?
XMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd_grad/tuple/control_dependency_1IdentityIMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd_grad/BiasAddGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd_grad/tuple/group_deps"/device:GPU:1*
_output_shapes	
:?*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd_grad/BiasAddGrad*
T0
?
JMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/Squeeze_grad/ShapeShape$My_GPU_1/CCN_1Conv_x0/convB11/conv1d"/device:GPU:1*
T0*
_output_shapes
:
?
LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/Squeeze_grad/ReshapeReshapeVMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd_grad/tuple/control_dependencyJMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/Squeeze_grad/Shape"/device:GPU:1*
T0*0
_output_shapes
:??????????
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/ShapeNShapeN/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_1"/device:GPU:1*
N* 
_output_shapes
::*
T0
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/Conv2DBackpropInputConv2DBackpropInputCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/ShapeN1My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/Squeeze_grad/Reshape"/device:GPU:1*
paddingSAME*
strides
*
T0*0
_output_shapes
:??????????
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/Conv2DBackpropFilterConv2DBackpropFilter/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDimsEMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/ShapeN:1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/Squeeze_grad/Reshape"/device:GPU:1*
paddingSAME*
strides
*
T0*(
_output_shapes
:??
?
MMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/tuple/group_depsNoOpR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/Conv2DBackpropFilterQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/Conv2DBackpropInput"/device:GPU:1
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/Conv2DBackpropInputN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/tuple/group_deps"/device:GPU:1*
T0*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/Conv2DBackpropInput*0
_output_shapes
:??????????
?
WMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/tuple/control_dependency_1IdentityQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/Conv2DBackpropFilterN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/tuple/group_deps"/device:GPU:1*
T0*(
_output_shapes
:??*d
_classZ
XVloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/Conv2DBackpropFilter
?
MMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_grad/ShapeShape"My_GPU_1/CCN_1Conv_x0/convA11/Relu"/device:GPU:1*
T0*
_output_shapes
:
?
OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_grad/ReshapeReshapeUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/tuple/control_dependencyMMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_grad/Shape"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_1_grad/ShapeConst"/device:GPU:1*
dtype0*!
valueB"         *
_output_shapes
:
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_1_grad/ReshapeReshapeWMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d_grad/tuple/control_dependency_1OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_1_grad/Shape"/device:GPU:1*
T0*$
_output_shapes
:??
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/Relu_grad/ReluGradReluGradOMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_grad/Reshape"My_GPU_1/CCN_1Conv_x0/convA11/Relu"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
My_GPU_1/gradients/AddN_12AddNUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Square_grad/Mul_1QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/conv1d/ExpandDims_1_grad/Reshape"/device:GPU:1*
N*
T0*h
_class^
\Zloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/Square_grad/Mul_1*$
_output_shapes
:??
?
IMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd_grad/BiasAddGradBiasAddGradCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/Relu_grad/ReluGrad"/device:GPU:1*
T0*
_output_shapes	
:?
?
NMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd_grad/tuple/group_depsNoOpJ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd_grad/BiasAddGradD^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/Relu_grad/ReluGrad"/device:GPU:1
?
VMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd_grad/tuple/control_dependencyIdentityCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/Relu_grad/ReluGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd_grad/tuple/group_deps"/device:GPU:1*V
_classL
JHloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/Relu_grad/ReluGrad*
T0*,
_output_shapes
:??????????
?
XMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd_grad/tuple/control_dependency_1IdentityIMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd_grad/BiasAddGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd_grad/tuple/group_deps"/device:GPU:1*
_output_shapes	
:?*
T0*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd_grad/BiasAddGrad
?
JMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/Squeeze_grad/ShapeShape$My_GPU_1/CCN_1Conv_x0/convA11/conv1d"/device:GPU:1*
T0*
_output_shapes
:
?
LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/Squeeze_grad/ReshapeReshapeVMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd_grad/tuple/control_dependencyJMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/Squeeze_grad/Shape"/device:GPU:1*
T0*0
_output_shapes
:??????????
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/ShapeNShapeN/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_1"/device:GPU:1*
N*
T0* 
_output_shapes
::
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/Conv2DBackpropInputConv2DBackpropInputCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/ShapeN1My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/Squeeze_grad/Reshape"/device:GPU:1*
paddingSAME*
strides
*
T0*0
_output_shapes
:??????????
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/Conv2DBackpropFilterConv2DBackpropFilter/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDimsEMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/ShapeN:1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/Squeeze_grad/Reshape"/device:GPU:1*(
_output_shapes
:??*
strides
*
T0*
paddingSAME
?
MMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/tuple/group_depsNoOpR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/Conv2DBackpropFilterQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/Conv2DBackpropInput"/device:GPU:1
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/Conv2DBackpropInputN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/tuple/group_deps"/device:GPU:1*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/Conv2DBackpropInput*0
_output_shapes
:??????????*
T0
?
WMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/tuple/control_dependency_1IdentityQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/Conv2DBackpropFilterN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/tuple/group_deps"/device:GPU:1*(
_output_shapes
:??*
T0*d
_classZ
XVloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/Conv2DBackpropFilter
?
MMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_grad/ShapeShape"My_GPU_1/CCN_1Conv_x0/convB20/Relu"/device:GPU:1*
_output_shapes
:*
T0
?
OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_grad/ReshapeReshapeUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/tuple/control_dependencyMMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_grad/Shape"/device:GPU:1*,
_output_shapes
:??????????*
T0
?
OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_1_grad/ShapeConst"/device:GPU:1*
_output_shapes
:*
dtype0*!
valueB"         
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_1_grad/ReshapeReshapeWMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d_grad/tuple/control_dependency_1OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_1_grad/Shape"/device:GPU:1*
T0*$
_output_shapes
:??
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/Relu_grad/ReluGradReluGradOMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_grad/Reshape"My_GPU_1/CCN_1Conv_x0/convB20/Relu"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
My_GPU_1/gradients/AddN_13AddNUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Square_grad/Mul_1QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/conv1d/ExpandDims_1_grad/Reshape"/device:GPU:1*h
_class^
\Zloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/Square_grad/Mul_1*$
_output_shapes
:??*
N*
T0
?
IMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd_grad/BiasAddGradBiasAddGradCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/Relu_grad/ReluGrad"/device:GPU:1*
_output_shapes	
:?*
T0
?
NMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd_grad/tuple/group_depsNoOpJ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd_grad/BiasAddGradD^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/Relu_grad/ReluGrad"/device:GPU:1
?
VMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd_grad/tuple/control_dependencyIdentityCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/Relu_grad/ReluGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd_grad/tuple/group_deps"/device:GPU:1*
T0*,
_output_shapes
:??????????*V
_classL
JHloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/Relu_grad/ReluGrad
?
XMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd_grad/tuple/control_dependency_1IdentityIMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd_grad/BiasAddGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd_grad/tuple/group_deps"/device:GPU:1*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?*
T0
?
JMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/Squeeze_grad/ShapeShape$My_GPU_1/CCN_1Conv_x0/convB20/conv1d"/device:GPU:1*
T0*
_output_shapes
:
?
LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/Squeeze_grad/ReshapeReshapeVMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd_grad/tuple/control_dependencyJMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/Squeeze_grad/Shape"/device:GPU:1*0
_output_shapes
:??????????*
T0
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/ShapeNShapeN/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_1"/device:GPU:1*
T0*
N* 
_output_shapes
::
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/Conv2DBackpropInputConv2DBackpropInputCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/ShapeN1My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/Squeeze_grad/Reshape"/device:GPU:1*0
_output_shapes
:??????????*
T0*
strides
*
paddingSAME
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/Conv2DBackpropFilterConv2DBackpropFilter/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDimsEMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/ShapeN:1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/Squeeze_grad/Reshape"/device:GPU:1*(
_output_shapes
:??*
paddingSAME*
T0*
strides

?
MMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/tuple/group_depsNoOpR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/Conv2DBackpropFilterQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/Conv2DBackpropInput"/device:GPU:1
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/Conv2DBackpropInputN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/tuple/group_deps"/device:GPU:1*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/Conv2DBackpropInput*0
_output_shapes
:??????????*
T0
?
WMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/tuple/control_dependency_1IdentityQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/Conv2DBackpropFilterN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/tuple/group_deps"/device:GPU:1*
T0*d
_classZ
XVloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/Conv2DBackpropFilter*(
_output_shapes
:??
?
MMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_grad/ShapeShape%My_GPU_1/CCN_1Conv_x0/poolB10/Squeeze"/device:GPU:1*
_output_shapes
:*
T0
?
OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_grad/ReshapeReshapeUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/tuple/control_dependencyMMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_grad/Shape"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_1_grad/ShapeConst"/device:GPU:1*!
valueB"         *
dtype0*
_output_shapes
:
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_1_grad/ReshapeReshapeWMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d_grad/tuple/control_dependency_1OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_1_grad/Shape"/device:GPU:1*$
_output_shapes
:??*
T0
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB10/Squeeze_grad/ShapeShapeMy_GPU_1/CCN_1Conv_x0/poolB10"/device:GPU:1*
_output_shapes
:*
T0
?
EMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB10/Squeeze_grad/ReshapeReshapeOMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_grad/ReshapeCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB10/Squeeze_grad/Shape"/device:GPU:1*0
_output_shapes
:??????????*
T0
?
My_GPU_1/gradients/AddN_14AddNUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Square_grad/Mul_1QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/conv1d/ExpandDims_1_grad/Reshape"/device:GPU:1*h
_class^
\Zloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/Square_grad/Mul_1*
N*
T0*$
_output_shapes
:??
?
AMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB10_grad/MaxPoolGradMaxPoolGrad(My_GPU_1/CCN_1Conv_x0/poolB10/ExpandDimsMy_GPU_1/CCN_1Conv_x0/poolB10EMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB10/Squeeze_grad/Reshape"/device:GPU:1*
ksize
*
strides
*0
_output_shapes
:??????????*
paddingSAME
?
FMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB10/ExpandDims_grad/ShapeShape"My_GPU_1/CCN_1Conv_x0/convB10/Relu"/device:GPU:1*
T0*
_output_shapes
:
?
HMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB10/ExpandDims_grad/ReshapeReshapeAMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB10_grad/MaxPoolGradFMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB10/ExpandDims_grad/Shape"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/Relu_grad/ReluGradReluGradHMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/poolB10/ExpandDims_grad/Reshape"My_GPU_1/CCN_1Conv_x0/convB10/Relu"/device:GPU:1*,
_output_shapes
:??????????*
T0
?
IMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd_grad/BiasAddGradBiasAddGradCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/Relu_grad/ReluGrad"/device:GPU:1*
_output_shapes	
:?*
T0
?
NMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd_grad/tuple/group_depsNoOpJ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd_grad/BiasAddGradD^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/Relu_grad/ReluGrad"/device:GPU:1
?
VMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd_grad/tuple/control_dependencyIdentityCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/Relu_grad/ReluGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd_grad/tuple/group_deps"/device:GPU:1*V
_classL
JHloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/Relu_grad/ReluGrad*,
_output_shapes
:??????????*
T0
?
XMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd_grad/tuple/control_dependency_1IdentityIMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd_grad/BiasAddGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd_grad/tuple/group_deps"/device:GPU:1*
_output_shapes	
:?*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd_grad/BiasAddGrad*
T0
?
JMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/Squeeze_grad/ShapeShape$My_GPU_1/CCN_1Conv_x0/convB10/conv1d"/device:GPU:1*
T0*
_output_shapes
:
?
LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/Squeeze_grad/ReshapeReshapeVMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd_grad/tuple/control_dependencyJMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/Squeeze_grad/Shape"/device:GPU:1*
T0*0
_output_shapes
:??????????
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/ShapeNShapeN/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_1"/device:GPU:1*
T0* 
_output_shapes
::*
N
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/Conv2DBackpropInputConv2DBackpropInputCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/ShapeN1My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/Squeeze_grad/Reshape"/device:GPU:1*
T0*
paddingSAME*
strides
*0
_output_shapes
:??????????
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/Conv2DBackpropFilterConv2DBackpropFilter/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDimsEMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/ShapeN:1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/Squeeze_grad/Reshape"/device:GPU:1*
T0*
paddingSAME*(
_output_shapes
:??*
strides

?
MMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/tuple/group_depsNoOpR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/Conv2DBackpropFilterQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/Conv2DBackpropInput"/device:GPU:1
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/Conv2DBackpropInputN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/tuple/group_deps"/device:GPU:1*0
_output_shapes
:??????????*
T0*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/Conv2DBackpropInput
?
WMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/tuple/control_dependency_1IdentityQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/Conv2DBackpropFilterN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/tuple/group_deps"/device:GPU:1*(
_output_shapes
:??*d
_classZ
XVloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/Conv2DBackpropFilter*
T0
?
MMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_grad/ShapeShape"My_GPU_1/CCN_1Conv_x0/convA10/Relu"/device:GPU:1*
T0*
_output_shapes
:
?
OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_grad/ReshapeReshapeUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/tuple/control_dependencyMMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_grad/Shape"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_1_grad/ShapeConst"/device:GPU:1*!
valueB"         *
dtype0*
_output_shapes
:
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_1_grad/ReshapeReshapeWMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d_grad/tuple/control_dependency_1OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_1_grad/Shape"/device:GPU:1*$
_output_shapes
:??*
T0
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/Relu_grad/ReluGradReluGradOMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_grad/Reshape"My_GPU_1/CCN_1Conv_x0/convA10/Relu"/device:GPU:1*
T0*,
_output_shapes
:??????????
?
My_GPU_1/gradients/AddN_15AddNUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Square_grad/Mul_1QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/conv1d/ExpandDims_1_grad/Reshape"/device:GPU:1*
N*$
_output_shapes
:??*h
_class^
\Zloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/Square_grad/Mul_1*
T0
?
IMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd_grad/BiasAddGradBiasAddGradCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/Relu_grad/ReluGrad"/device:GPU:1*
_output_shapes	
:?*
T0
?
NMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd_grad/tuple/group_depsNoOpJ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd_grad/BiasAddGradD^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/Relu_grad/ReluGrad"/device:GPU:1
?
VMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd_grad/tuple/control_dependencyIdentityCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/Relu_grad/ReluGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd_grad/tuple/group_deps"/device:GPU:1*,
_output_shapes
:??????????*V
_classL
JHloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/Relu_grad/ReluGrad*
T0
?
XMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd_grad/tuple/control_dependency_1IdentityIMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd_grad/BiasAddGradO^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd_grad/tuple/group_deps"/device:GPU:1*\
_classR
PNloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?*
T0
?
JMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/Squeeze_grad/ShapeShape$My_GPU_1/CCN_1Conv_x0/convA10/conv1d"/device:GPU:1*
_output_shapes
:*
T0
?
LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/Squeeze_grad/ReshapeReshapeVMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd_grad/tuple/control_dependencyJMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/Squeeze_grad/Shape"/device:GPU:1*
T0*0
_output_shapes
:??????????
?
CMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/ShapeNShapeN/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims1My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims_1"/device:GPU:1*
T0*
N* 
_output_shapes
::
?
PMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/Conv2DBackpropInputConv2DBackpropInputCMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/ShapeN1My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims_1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/Squeeze_grad/Reshape"/device:GPU:1*
strides
*
paddingSAME*/
_output_shapes
:?????????*
T0
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/Conv2DBackpropFilterConv2DBackpropFilter/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDimsEMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/ShapeN:1LMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/Squeeze_grad/Reshape"/device:GPU:1*
paddingSAME*
strides
*
T0*'
_output_shapes
:?
?
MMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/tuple/group_depsNoOpR^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/Conv2DBackpropFilterQ^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/Conv2DBackpropInput"/device:GPU:1
?
UMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/tuple/control_dependencyIdentityPMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/Conv2DBackpropInputN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/tuple/group_deps"/device:GPU:1*/
_output_shapes
:?????????*
T0*c
_classY
WUloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/Conv2DBackpropInput
?
WMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/tuple/control_dependency_1IdentityQMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/Conv2DBackpropFilterN^My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/tuple/group_deps"/device:GPU:1*'
_output_shapes
:?*d
_classZ
XVloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/Conv2DBackpropFilter*
T0
?
OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims_1_grad/ShapeConst"/device:GPU:1*
dtype0*!
valueB"         *
_output_shapes
:
?
QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims_1_grad/ReshapeReshapeWMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d_grad/tuple/control_dependency_1OMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims_1_grad/Shape"/device:GPU:1*#
_output_shapes
:?*
T0
?
My_GPU_1/gradients/AddN_16AddNUMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Square_grad/Mul_1QMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/conv1d/ExpandDims_1_grad/Reshape"/device:GPU:1*#
_output_shapes
:?*h
_class^
\Zloc:@My_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/Square_grad/Mul_1*
T0*
N
t
 My_GPU_1/clip_by_value/Minimum/yConst"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
My_GPU_1/clip_by_value/MinimumMinimumMy_GPU_1/gradients/AddN_16 My_GPU_1/clip_by_value/Minimum/y"/device:GPU:1*#
_output_shapes
:?*
T0
l
My_GPU_1/clip_by_value/yConst"/device:GPU:1*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
My_GPU_1/clip_by_valueMaximumMy_GPU_1/clip_by_value/MinimumMy_GPU_1/clip_by_value/y"/device:GPU:1*
T0*#
_output_shapes
:?
v
"My_GPU_1/clip_by_value_1/Minimum/yConst"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
 My_GPU_1/clip_by_value_1/MinimumMinimumXMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA10/BiasAdd_grad/tuple/control_dependency_1"My_GPU_1/clip_by_value_1/Minimum/y"/device:GPU:1*
T0*
_output_shapes	
:?
n
My_GPU_1/clip_by_value_1/yConst"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
My_GPU_1/clip_by_value_1Maximum My_GPU_1/clip_by_value_1/MinimumMy_GPU_1/clip_by_value_1/y"/device:GPU:1*
T0*
_output_shapes	
:?
v
"My_GPU_1/clip_by_value_2/Minimum/yConst"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
 My_GPU_1/clip_by_value_2/MinimumMinimumMy_GPU_1/gradients/AddN_15"My_GPU_1/clip_by_value_2/Minimum/y"/device:GPU:1*
T0*$
_output_shapes
:??
n
My_GPU_1/clip_by_value_2/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
My_GPU_1/clip_by_value_2Maximum My_GPU_1/clip_by_value_2/MinimumMy_GPU_1/clip_by_value_2/y"/device:GPU:1*
T0*$
_output_shapes
:??
v
"My_GPU_1/clip_by_value_3/Minimum/yConst"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
 My_GPU_1/clip_by_value_3/MinimumMinimumXMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB10/BiasAdd_grad/tuple/control_dependency_1"My_GPU_1/clip_by_value_3/Minimum/y"/device:GPU:1*
T0*
_output_shapes	
:?
n
My_GPU_1/clip_by_value_3/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
My_GPU_1/clip_by_value_3Maximum My_GPU_1/clip_by_value_3/MinimumMy_GPU_1/clip_by_value_3/y"/device:GPU:1*
_output_shapes	
:?*
T0
v
"My_GPU_1/clip_by_value_4/Minimum/yConst"/device:GPU:1*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
 My_GPU_1/clip_by_value_4/MinimumMinimumMy_GPU_1/gradients/AddN_14"My_GPU_1/clip_by_value_4/Minimum/y"/device:GPU:1*$
_output_shapes
:??*
T0
n
My_GPU_1/clip_by_value_4/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
My_GPU_1/clip_by_value_4Maximum My_GPU_1/clip_by_value_4/MinimumMy_GPU_1/clip_by_value_4/y"/device:GPU:1*$
_output_shapes
:??*
T0
v
"My_GPU_1/clip_by_value_5/Minimum/yConst"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
 My_GPU_1/clip_by_value_5/MinimumMinimumXMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB20/BiasAdd_grad/tuple/control_dependency_1"My_GPU_1/clip_by_value_5/Minimum/y"/device:GPU:1*
_output_shapes	
:?*
T0
n
My_GPU_1/clip_by_value_5/yConst"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
My_GPU_1/clip_by_value_5Maximum My_GPU_1/clip_by_value_5/MinimumMy_GPU_1/clip_by_value_5/y"/device:GPU:1*
T0*
_output_shapes	
:?
v
"My_GPU_1/clip_by_value_6/Minimum/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
 My_GPU_1/clip_by_value_6/MinimumMinimumMy_GPU_1/gradients/AddN_13"My_GPU_1/clip_by_value_6/Minimum/y"/device:GPU:1*$
_output_shapes
:??*
T0
n
My_GPU_1/clip_by_value_6/yConst"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
My_GPU_1/clip_by_value_6Maximum My_GPU_1/clip_by_value_6/MinimumMy_GPU_1/clip_by_value_6/y"/device:GPU:1*
T0*$
_output_shapes
:??
v
"My_GPU_1/clip_by_value_7/Minimum/yConst"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
 My_GPU_1/clip_by_value_7/MinimumMinimumXMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convA11/BiasAdd_grad/tuple/control_dependency_1"My_GPU_1/clip_by_value_7/Minimum/y"/device:GPU:1*
_output_shapes	
:?*
T0
n
My_GPU_1/clip_by_value_7/yConst"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
My_GPU_1/clip_by_value_7Maximum My_GPU_1/clip_by_value_7/MinimumMy_GPU_1/clip_by_value_7/y"/device:GPU:1*
T0*
_output_shapes	
:?
v
"My_GPU_1/clip_by_value_8/Minimum/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
 My_GPU_1/clip_by_value_8/MinimumMinimumMy_GPU_1/gradients/AddN_12"My_GPU_1/clip_by_value_8/Minimum/y"/device:GPU:1*
T0*$
_output_shapes
:??
n
My_GPU_1/clip_by_value_8/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
My_GPU_1/clip_by_value_8Maximum My_GPU_1/clip_by_value_8/MinimumMy_GPU_1/clip_by_value_8/y"/device:GPU:1*
T0*$
_output_shapes
:??
v
"My_GPU_1/clip_by_value_9/Minimum/yConst"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
 My_GPU_1/clip_by_value_9/MinimumMinimumXMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB11/BiasAdd_grad/tuple/control_dependency_1"My_GPU_1/clip_by_value_9/Minimum/y"/device:GPU:1*
_output_shapes	
:?*
T0
n
My_GPU_1/clip_by_value_9/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
My_GPU_1/clip_by_value_9Maximum My_GPU_1/clip_by_value_9/MinimumMy_GPU_1/clip_by_value_9/y"/device:GPU:1*
_output_shapes	
:?*
T0
w
#My_GPU_1/clip_by_value_10/Minimum/yConst"/device:GPU:1*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
!My_GPU_1/clip_by_value_10/MinimumMinimumMy_GPU_1/gradients/AddN_11#My_GPU_1/clip_by_value_10/Minimum/y"/device:GPU:1*
T0*$
_output_shapes
:??
o
My_GPU_1/clip_by_value_10/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
My_GPU_1/clip_by_value_10Maximum!My_GPU_1/clip_by_value_10/MinimumMy_GPU_1/clip_by_value_10/y"/device:GPU:1*$
_output_shapes
:??*
T0
w
#My_GPU_1/clip_by_value_11/Minimum/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
!My_GPU_1/clip_by_value_11/MinimumMinimumXMy_GPU_1/gradients/My_GPU_1/CCN_1Conv_x0/convB21/BiasAdd_grad/tuple/control_dependency_1#My_GPU_1/clip_by_value_11/Minimum/y"/device:GPU:1*
_output_shapes	
:?*
T0
o
My_GPU_1/clip_by_value_11/yConst"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
My_GPU_1/clip_by_value_11Maximum!My_GPU_1/clip_by_value_11/MinimumMy_GPU_1/clip_by_value_11/y"/device:GPU:1*
_output_shapes	
:?*
T0
w
#My_GPU_1/clip_by_value_12/Minimum/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
!My_GPU_1/clip_by_value_12/MinimumMinimum]My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/sub_grad/tuple/control_dependency#My_GPU_1/clip_by_value_12/Minimum/y"/device:GPU:1*
T0*
_output_shapes	
:?
o
My_GPU_1/clip_by_value_12/yConst"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
My_GPU_1/clip_by_value_12Maximum!My_GPU_1/clip_by_value_12/MinimumMy_GPU_1/clip_by_value_12/y"/device:GPU:1*
_output_shapes	
:?*
T0
w
#My_GPU_1/clip_by_value_13/Minimum/yConst"/device:GPU:1*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
!My_GPU_1/clip_by_value_13/MinimumMinimum_My_GPU_1/gradients/My_GPU_1/Conv_out__/Conv_out__/batchnorm/mul_grad/tuple/control_dependency_1#My_GPU_1/clip_by_value_13/Minimum/y"/device:GPU:1*
_output_shapes	
:?*
T0
o
My_GPU_1/clip_by_value_13/yConst"/device:GPU:1*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
My_GPU_1/clip_by_value_13Maximum!My_GPU_1/clip_by_value_13/MinimumMy_GPU_1/clip_by_value_13/y"/device:GPU:1*
T0*
_output_shapes	
:?
w
#My_GPU_1/clip_by_value_14/Minimum/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
!My_GPU_1/clip_by_value_14/MinimumMinimumMy_GPU_1/gradients/AddN_1#My_GPU_1/clip_by_value_14/Minimum/y"/device:GPU:1*
_output_shapes
:	?*
T0
o
My_GPU_1/clip_by_value_14/yConst"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
My_GPU_1/clip_by_value_14Maximum!My_GPU_1/clip_by_value_14/MinimumMy_GPU_1/clip_by_value_14/y"/device:GPU:1*
T0*
_output_shapes
:	?
w
#My_GPU_1/clip_by_value_15/Minimum/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
!My_GPU_1/clip_by_value_15/MinimumMinimum_My_GPU_1/gradients/My_GPU_1/Reconstruction_Output/dense/BiasAdd_grad/tuple/control_dependency_1#My_GPU_1/clip_by_value_15/Minimum/y"/device:GPU:1*
T0*
_output_shapes
:
o
My_GPU_1/clip_by_value_15/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
My_GPU_1/clip_by_value_15Maximum!My_GPU_1/clip_by_value_15/MinimumMy_GPU_1/clip_by_value_15/y"/device:GPU:1*
T0*
_output_shapes
:
w
#My_GPU_1/clip_by_value_16/Minimum/yConst"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
!My_GPU_1/clip_by_value_16/MinimumMinimumMy_GPU_1/gradients/AddN_7#My_GPU_1/clip_by_value_16/Minimum/y"/device:GPU:1*
_output_shapes
:	? *
T0
o
My_GPU_1/clip_by_value_16/yConst"/device:GPU:1*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
My_GPU_1/clip_by_value_16Maximum!My_GPU_1/clip_by_value_16/MinimumMy_GPU_1/clip_by_value_16/y"/device:GPU:1*
_output_shapes
:	? *
T0
w
#My_GPU_1/clip_by_value_17/Minimum/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
!My_GPU_1/clip_by_value_17/MinimumMinimumIMy_GPU_1/gradients/My_GPU_1/dense/BiasAdd_grad/tuple/control_dependency_1#My_GPU_1/clip_by_value_17/Minimum/y"/device:GPU:1*
_output_shapes
: *
T0
o
My_GPU_1/clip_by_value_17/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
My_GPU_1/clip_by_value_17Maximum!My_GPU_1/clip_by_value_17/MinimumMy_GPU_1/clip_by_value_17/y"/device:GPU:1*
T0*
_output_shapes
: 
w
#My_GPU_1/clip_by_value_18/Minimum/yConst"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
!My_GPU_1/clip_by_value_18/MinimumMinimumMy_GPU_1/gradients/AddN_6#My_GPU_1/clip_by_value_18/Minimum/y"/device:GPU:1*
_output_shapes

:  *
T0
o
My_GPU_1/clip_by_value_18/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
My_GPU_1/clip_by_value_18Maximum!My_GPU_1/clip_by_value_18/MinimumMy_GPU_1/clip_by_value_18/y"/device:GPU:1*
T0*
_output_shapes

:  
w
#My_GPU_1/clip_by_value_19/Minimum/yConst"/device:GPU:1*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
!My_GPU_1/clip_by_value_19/MinimumMinimum[My_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/dense/BiasAdd_grad/tuple/control_dependency_1#My_GPU_1/clip_by_value_19/Minimum/y"/device:GPU:1*
T0*
_output_shapes
: 
o
My_GPU_1/clip_by_value_19/yConst"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
My_GPU_1/clip_by_value_19Maximum!My_GPU_1/clip_by_value_19/MinimumMy_GPU_1/clip_by_value_19/y"/device:GPU:1*
T0*
_output_shapes
: 
w
#My_GPU_1/clip_by_value_20/Minimum/yConst"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
!My_GPU_1/clip_by_value_20/MinimumMinimumkMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/sub_grad/tuple/control_dependency#My_GPU_1/clip_by_value_20/Minimum/y"/device:GPU:1*
_output_shapes
: *
T0
o
My_GPU_1/clip_by_value_20/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
My_GPU_1/clip_by_value_20Maximum!My_GPU_1/clip_by_value_20/MinimumMy_GPU_1/clip_by_value_20/y"/device:GPU:1*
T0*
_output_shapes
: 
w
#My_GPU_1/clip_by_value_21/Minimum/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
!My_GPU_1/clip_by_value_21/MinimumMinimummMy_GPU_1/gradients/My_GPU_1/FCU_muiltDense_x0/FCU_muiltDense_x0/batchnorm/mul_grad/tuple/control_dependency_1#My_GPU_1/clip_by_value_21/Minimum/y"/device:GPU:1*
T0*
_output_shapes
: 
o
My_GPU_1/clip_by_value_21/yConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
My_GPU_1/clip_by_value_21Maximum!My_GPU_1/clip_by_value_21/MinimumMy_GPU_1/clip_by_value_21/y"/device:GPU:1*
T0*
_output_shapes
: 
w
#My_GPU_1/clip_by_value_22/Minimum/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
!My_GPU_1/clip_by_value_22/MinimumMinimumMy_GPU_1/gradients/AddN_2#My_GPU_1/clip_by_value_22/Minimum/y"/device:GPU:1*
_output_shapes

: *
T0
o
My_GPU_1/clip_by_value_22/yConst"/device:GPU:1*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
My_GPU_1/clip_by_value_22Maximum!My_GPU_1/clip_by_value_22/MinimumMy_GPU_1/clip_by_value_22/y"/device:GPU:1*
_output_shapes

: *
T0
w
#My_GPU_1/clip_by_value_23/Minimum/yConst"/device:GPU:1*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
!My_GPU_1/clip_by_value_23/MinimumMinimumQMy_GPU_1/gradients/My_GPU_1/Output_/dense/BiasAdd_grad/tuple/control_dependency_1#My_GPU_1/clip_by_value_23/Minimum/y"/device:GPU:1*
T0*
_output_shapes
:
o
My_GPU_1/clip_by_value_23/yConst"/device:GPU:1*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
My_GPU_1/clip_by_value_23Maximum!My_GPU_1/clip_by_value_23/MinimumMy_GPU_1/clip_by_value_23/y"/device:GPU:1*
_output_shapes
:*
T0
_
ExpandDims/dimConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
?

ExpandDims
ExpandDimsMy_GPU_1/clip_by_valueExpandDims/dim"/device:CPU:0*
T0*'
_output_shapes
:?
b
concat/concat_dimConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
f
concat/concatIdentity
ExpandDims"/device:CPU:0*'
_output_shapes
:?*
T0
g
Mean/reduction_indicesConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
p
MeanMeanconcat/concatMean/reduction_indices"/device:CPU:0*#
_output_shapes
:?*
T0
a
ExpandDims_1/dimConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 

ExpandDims_1
ExpandDimsMy_GPU_1/clip_by_value_1ExpandDims_1/dim"/device:CPU:0*
_output_shapes
:	?*
T0
d
concat_1/concat_dimConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0
b
concat_1/concatIdentityExpandDims_1"/device:CPU:0*
T0*
_output_shapes
:	?
i
Mean_1/reduction_indicesConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
n
Mean_1Meanconcat_1/concatMean_1/reduction_indices"/device:CPU:0*
_output_shapes	
:?*
T0
a
ExpandDims_2/dimConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
?
ExpandDims_2
ExpandDimsMy_GPU_1/clip_by_value_2ExpandDims_2/dim"/device:CPU:0*(
_output_shapes
:??*
T0
d
concat_2/concat_dimConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
k
concat_2/concatIdentityExpandDims_2"/device:CPU:0*(
_output_shapes
:??*
T0
i
Mean_2/reduction_indicesConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
w
Mean_2Meanconcat_2/concatMean_2/reduction_indices"/device:CPU:0*
T0*$
_output_shapes
:??
a
ExpandDims_3/dimConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0

ExpandDims_3
ExpandDimsMy_GPU_1/clip_by_value_3ExpandDims_3/dim"/device:CPU:0*
T0*
_output_shapes
:	?
d
concat_3/concat_dimConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
b
concat_3/concatIdentityExpandDims_3"/device:CPU:0*
T0*
_output_shapes
:	?
i
Mean_3/reduction_indicesConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
n
Mean_3Meanconcat_3/concatMean_3/reduction_indices"/device:CPU:0*
_output_shapes	
:?*
T0
a
ExpandDims_4/dimConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
?
ExpandDims_4
ExpandDimsMy_GPU_1/clip_by_value_4ExpandDims_4/dim"/device:CPU:0*
T0*(
_output_shapes
:??
d
concat_4/concat_dimConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
k
concat_4/concatIdentityExpandDims_4"/device:CPU:0*
T0*(
_output_shapes
:??
i
Mean_4/reduction_indicesConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
w
Mean_4Meanconcat_4/concatMean_4/reduction_indices"/device:CPU:0*
T0*$
_output_shapes
:??
a
ExpandDims_5/dimConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 

ExpandDims_5
ExpandDimsMy_GPU_1/clip_by_value_5ExpandDims_5/dim"/device:CPU:0*
T0*
_output_shapes
:	?
d
concat_5/concat_dimConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
b
concat_5/concatIdentityExpandDims_5"/device:CPU:0*
T0*
_output_shapes
:	?
i
Mean_5/reduction_indicesConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
n
Mean_5Meanconcat_5/concatMean_5/reduction_indices"/device:CPU:0*
T0*
_output_shapes	
:?
a
ExpandDims_6/dimConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
ExpandDims_6
ExpandDimsMy_GPU_1/clip_by_value_6ExpandDims_6/dim"/device:CPU:0*(
_output_shapes
:??*
T0
d
concat_6/concat_dimConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
k
concat_6/concatIdentityExpandDims_6"/device:CPU:0*
T0*(
_output_shapes
:??
i
Mean_6/reduction_indicesConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
w
Mean_6Meanconcat_6/concatMean_6/reduction_indices"/device:CPU:0*
T0*$
_output_shapes
:??
a
ExpandDims_7/dimConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 

ExpandDims_7
ExpandDimsMy_GPU_1/clip_by_value_7ExpandDims_7/dim"/device:CPU:0*
T0*
_output_shapes
:	?
d
concat_7/concat_dimConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
b
concat_7/concatIdentityExpandDims_7"/device:CPU:0*
_output_shapes
:	?*
T0
i
Mean_7/reduction_indicesConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0
n
Mean_7Meanconcat_7/concatMean_7/reduction_indices"/device:CPU:0*
T0*
_output_shapes	
:?
a
ExpandDims_8/dimConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
?
ExpandDims_8
ExpandDimsMy_GPU_1/clip_by_value_8ExpandDims_8/dim"/device:CPU:0*
T0*(
_output_shapes
:??
d
concat_8/concat_dimConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
k
concat_8/concatIdentityExpandDims_8"/device:CPU:0*(
_output_shapes
:??*
T0
i
Mean_8/reduction_indicesConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
w
Mean_8Meanconcat_8/concatMean_8/reduction_indices"/device:CPU:0*$
_output_shapes
:??*
T0
a
ExpandDims_9/dimConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

ExpandDims_9
ExpandDimsMy_GPU_1/clip_by_value_9ExpandDims_9/dim"/device:CPU:0*
T0*
_output_shapes
:	?
d
concat_9/concat_dimConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0
b
concat_9/concatIdentityExpandDims_9"/device:CPU:0*
_output_shapes
:	?*
T0
i
Mean_9/reduction_indicesConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
n
Mean_9Meanconcat_9/concatMean_9/reduction_indices"/device:CPU:0*
T0*
_output_shapes	
:?
b
ExpandDims_10/dimConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0
?
ExpandDims_10
ExpandDimsMy_GPU_1/clip_by_value_10ExpandDims_10/dim"/device:CPU:0*
T0*(
_output_shapes
:??
e
concat_10/concat_dimConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
m
concat_10/concatIdentityExpandDims_10"/device:CPU:0*(
_output_shapes
:??*
T0
j
Mean_10/reduction_indicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
z
Mean_10Meanconcat_10/concatMean_10/reduction_indices"/device:CPU:0*$
_output_shapes
:??*
T0
b
ExpandDims_11/dimConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0
?
ExpandDims_11
ExpandDimsMy_GPU_1/clip_by_value_11ExpandDims_11/dim"/device:CPU:0*
T0*
_output_shapes
:	?
e
concat_11/concat_dimConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
d
concat_11/concatIdentityExpandDims_11"/device:CPU:0*
_output_shapes
:	?*
T0
j
Mean_11/reduction_indicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
q
Mean_11Meanconcat_11/concatMean_11/reduction_indices"/device:CPU:0*
_output_shapes	
:?*
T0
b
ExpandDims_12/dimConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
?
ExpandDims_12
ExpandDimsMy_GPU_1/clip_by_value_12ExpandDims_12/dim"/device:CPU:0*
_output_shapes
:	?*
T0
e
concat_12/concat_dimConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
d
concat_12/concatIdentityExpandDims_12"/device:CPU:0*
_output_shapes
:	?*
T0
j
Mean_12/reduction_indicesConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
q
Mean_12Meanconcat_12/concatMean_12/reduction_indices"/device:CPU:0*
_output_shapes	
:?*
T0
b
ExpandDims_13/dimConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
?
ExpandDims_13
ExpandDimsMy_GPU_1/clip_by_value_13ExpandDims_13/dim"/device:CPU:0*
T0*
_output_shapes
:	?
e
concat_13/concat_dimConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
d
concat_13/concatIdentityExpandDims_13"/device:CPU:0*
T0*
_output_shapes
:	?
j
Mean_13/reduction_indicesConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
q
Mean_13Meanconcat_13/concatMean_13/reduction_indices"/device:CPU:0*
T0*
_output_shapes	
:?
b
ExpandDims_14/dimConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
ExpandDims_14
ExpandDimsMy_GPU_1/clip_by_value_14ExpandDims_14/dim"/device:CPU:0*
T0*#
_output_shapes
:?
e
concat_14/concat_dimConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
h
concat_14/concatIdentityExpandDims_14"/device:CPU:0*#
_output_shapes
:?*
T0
j
Mean_14/reduction_indicesConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
u
Mean_14Meanconcat_14/concatMean_14/reduction_indices"/device:CPU:0*
_output_shapes
:	?*
T0
b
ExpandDims_15/dimConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0
?
ExpandDims_15
ExpandDimsMy_GPU_1/clip_by_value_15ExpandDims_15/dim"/device:CPU:0*
_output_shapes

:*
T0
e
concat_15/concat_dimConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
c
concat_15/concatIdentityExpandDims_15"/device:CPU:0*
T0*
_output_shapes

:
j
Mean_15/reduction_indicesConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
p
Mean_15Meanconcat_15/concatMean_15/reduction_indices"/device:CPU:0*
T0*
_output_shapes
:
b
ExpandDims_16/dimConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
?
ExpandDims_16
ExpandDimsMy_GPU_1/clip_by_value_16ExpandDims_16/dim"/device:CPU:0*
T0*#
_output_shapes
:? 
e
concat_16/concat_dimConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
h
concat_16/concatIdentityExpandDims_16"/device:CPU:0*#
_output_shapes
:? *
T0
j
Mean_16/reduction_indicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
u
Mean_16Meanconcat_16/concatMean_16/reduction_indices"/device:CPU:0*
_output_shapes
:	? *
T0
b
ExpandDims_17/dimConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
?
ExpandDims_17
ExpandDimsMy_GPU_1/clip_by_value_17ExpandDims_17/dim"/device:CPU:0*
_output_shapes

: *
T0
e
concat_17/concat_dimConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
c
concat_17/concatIdentityExpandDims_17"/device:CPU:0*
T0*
_output_shapes

: 
j
Mean_17/reduction_indicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
p
Mean_17Meanconcat_17/concatMean_17/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: 
b
ExpandDims_18/dimConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
?
ExpandDims_18
ExpandDimsMy_GPU_1/clip_by_value_18ExpandDims_18/dim"/device:CPU:0*"
_output_shapes
:  *
T0
e
concat_18/concat_dimConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
g
concat_18/concatIdentityExpandDims_18"/device:CPU:0*
T0*"
_output_shapes
:  
j
Mean_18/reduction_indicesConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
t
Mean_18Meanconcat_18/concatMean_18/reduction_indices"/device:CPU:0*
_output_shapes

:  *
T0
b
ExpandDims_19/dimConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
?
ExpandDims_19
ExpandDimsMy_GPU_1/clip_by_value_19ExpandDims_19/dim"/device:CPU:0*
_output_shapes

: *
T0
e
concat_19/concat_dimConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
c
concat_19/concatIdentityExpandDims_19"/device:CPU:0*
_output_shapes

: *
T0
j
Mean_19/reduction_indicesConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
p
Mean_19Meanconcat_19/concatMean_19/reduction_indices"/device:CPU:0*
_output_shapes
: *
T0
b
ExpandDims_20/dimConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
?
ExpandDims_20
ExpandDimsMy_GPU_1/clip_by_value_20ExpandDims_20/dim"/device:CPU:0*
_output_shapes

: *
T0
e
concat_20/concat_dimConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
c
concat_20/concatIdentityExpandDims_20"/device:CPU:0*
T0*
_output_shapes

: 
j
Mean_20/reduction_indicesConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
p
Mean_20Meanconcat_20/concatMean_20/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: 
b
ExpandDims_21/dimConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
?
ExpandDims_21
ExpandDimsMy_GPU_1/clip_by_value_21ExpandDims_21/dim"/device:CPU:0*
T0*
_output_shapes

: 
e
concat_21/concat_dimConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
c
concat_21/concatIdentityExpandDims_21"/device:CPU:0*
T0*
_output_shapes

: 
j
Mean_21/reduction_indicesConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
p
Mean_21Meanconcat_21/concatMean_21/reduction_indices"/device:CPU:0*
_output_shapes
: *
T0
b
ExpandDims_22/dimConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
?
ExpandDims_22
ExpandDimsMy_GPU_1/clip_by_value_22ExpandDims_22/dim"/device:CPU:0*
T0*"
_output_shapes
: 
e
concat_22/concat_dimConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
g
concat_22/concatIdentityExpandDims_22"/device:CPU:0*"
_output_shapes
: *
T0
j
Mean_22/reduction_indicesConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
t
Mean_22Meanconcat_22/concatMean_22/reduction_indices"/device:CPU:0*
T0*
_output_shapes

: 
b
ExpandDims_23/dimConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
?
ExpandDims_23
ExpandDimsMy_GPU_1/clip_by_value_23ExpandDims_23/dim"/device:CPU:0*
T0*
_output_shapes

:
e
concat_23/concat_dimConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
c
concat_23/concatIdentityExpandDims_23"/device:CPU:0*
_output_shapes

:*
T0
j
Mean_23/reduction_indicesConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
p
Mean_23Meanconcat_23/concatMean_23/reduction_indices"/device:CPU:0*
T0*
_output_shapes
:
?
beta1_power/initial_valueConst"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
valueB
 *fff?*
_output_shapes
: *
dtype0
?
beta1_power
VariableV2"/device:GPU:1*
shape: *
_output_shapes
: *
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes
: 
?
beta1_power/readIdentitybeta1_power"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes
: *
T0
?
beta2_power/initial_valueConst"/device:GPU:1*
valueB
 *w??*
dtype0*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
beta2_power
VariableV2"/device:GPU:1*
dtype0*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
shape: 
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value"/device:GPU:1*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0
?
beta2_power/readIdentitybeta2_power"/device:GPU:1*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0
?
BCCN_1Conv_x0/convA10/kernel/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
_output_shapes
:*!
valueB"         *
dtype0
?
8CCN_1Conv_x0/convA10/kernel/Adam/Initializer/zeros/ConstConst"/device:GPU:1*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
dtype0*
valueB
 *    
?
2CCN_1Conv_x0/convA10/kernel/Adam/Initializer/zerosFillBCCN_1Conv_x0/convA10/kernel/Adam/Initializer/zeros/shape_as_tensor8CCN_1Conv_x0/convA10/kernel/Adam/Initializer/zeros/Const"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?
?
 CCN_1Conv_x0/convA10/kernel/Adam
VariableV2"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?*
dtype0*
shape:?
?
'CCN_1Conv_x0/convA10/kernel/Adam/AssignAssign CCN_1Conv_x0/convA10/kernel/Adam2CCN_1Conv_x0/convA10/kernel/Adam/Initializer/zeros"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0*#
_output_shapes
:?
?
%CCN_1Conv_x0/convA10/kernel/Adam/readIdentity CCN_1Conv_x0/convA10/kernel/Adam"/device:GPU:1*
T0*#
_output_shapes
:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
DCCN_1Conv_x0/convA10/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*!
valueB"         *
_output_shapes
:*
dtype0
?
:CCN_1Conv_x0/convA10/kernel/Adam_1/Initializer/zeros/ConstConst"/device:GPU:1*
valueB
 *    *.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
dtype0*
_output_shapes
: 
?
4CCN_1Conv_x0/convA10/kernel/Adam_1/Initializer/zerosFillDCCN_1Conv_x0/convA10/kernel/Adam_1/Initializer/zeros/shape_as_tensor:CCN_1Conv_x0/convA10/kernel/Adam_1/Initializer/zeros/Const"/device:GPU:1*
T0*#
_output_shapes
:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
"CCN_1Conv_x0/convA10/kernel/Adam_1
VariableV2"/device:GPU:1*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?*
shape:?
?
)CCN_1Conv_x0/convA10/kernel/Adam_1/AssignAssign"CCN_1Conv_x0/convA10/kernel/Adam_14CCN_1Conv_x0/convA10/kernel/Adam_1/Initializer/zeros"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?*
T0
?
'CCN_1Conv_x0/convA10/kernel/Adam_1/readIdentity"CCN_1Conv_x0/convA10/kernel/Adam_1"/device:GPU:1*#
_output_shapes
:?*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
0CCN_1Conv_x0/convA10/bias/Adam/Initializer/zerosConst"/device:GPU:1*
_output_shapes	
:?*
valueB?*    *
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
CCN_1Conv_x0/convA10/bias/Adam
VariableV2"/device:GPU:1*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
shape:?*
_output_shapes	
:?
?
%CCN_1Conv_x0/convA10/bias/Adam/AssignAssignCCN_1Conv_x0/convA10/bias/Adam0CCN_1Conv_x0/convA10/bias/Adam/Initializer/zeros"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0*
_output_shapes	
:?
?
#CCN_1Conv_x0/convA10/bias/Adam/readIdentityCCN_1Conv_x0/convA10/bias/Adam"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?
?
2CCN_1Conv_x0/convA10/bias/Adam_1/Initializer/zerosConst"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
dtype0*
valueB?*    *
_output_shapes	
:?
?
 CCN_1Conv_x0/convA10/bias/Adam_1
VariableV2"/device:GPU:1*
_output_shapes	
:?*
shape:?*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
dtype0
?
'CCN_1Conv_x0/convA10/bias/Adam_1/AssignAssign CCN_1Conv_x0/convA10/bias/Adam_12CCN_1Conv_x0/convA10/bias/Adam_1/Initializer/zeros"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0
?
%CCN_1Conv_x0/convA10/bias/Adam_1/readIdentity CCN_1Conv_x0/convA10/bias/Adam_1"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
BCCN_1Conv_x0/convB10/kernel/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*!
valueB"         *
dtype0*
_output_shapes
:*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel
?
8CCN_1Conv_x0/convB10/kernel/Adam/Initializer/zeros/ConstConst"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
?
2CCN_1Conv_x0/convB10/kernel/Adam/Initializer/zerosFillBCCN_1Conv_x0/convB10/kernel/Adam/Initializer/zeros/shape_as_tensor8CCN_1Conv_x0/convB10/kernel/Adam/Initializer/zeros/Const"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??
?
 CCN_1Conv_x0/convB10/kernel/Adam
VariableV2"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
dtype0*$
_output_shapes
:??*
shape:??
?
'CCN_1Conv_x0/convB10/kernel/Adam/AssignAssign CCN_1Conv_x0/convB10/kernel/Adam2CCN_1Conv_x0/convB10/kernel/Adam/Initializer/zeros"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??
?
%CCN_1Conv_x0/convB10/kernel/Adam/readIdentity CCN_1Conv_x0/convB10/kernel/Adam"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0
?
DCCN_1Conv_x0/convB10/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
_output_shapes
:*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*!
valueB"         
?
:CCN_1Conv_x0/convB10/kernel/Adam_1/Initializer/zeros/ConstConst"/device:GPU:1*
dtype0*
valueB
 *    *.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
_output_shapes
: 
?
4CCN_1Conv_x0/convB10/kernel/Adam_1/Initializer/zerosFillDCCN_1Conv_x0/convB10/kernel/Adam_1/Initializer/zeros/shape_as_tensor:CCN_1Conv_x0/convB10/kernel/Adam_1/Initializer/zeros/Const"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0
?
"CCN_1Conv_x0/convB10/kernel/Adam_1
VariableV2"/device:GPU:1*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
shape:??*$
_output_shapes
:??
?
)CCN_1Conv_x0/convB10/kernel/Adam_1/AssignAssign"CCN_1Conv_x0/convB10/kernel/Adam_14CCN_1Conv_x0/convB10/kernel/Adam_1/Initializer/zeros"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0*$
_output_shapes
:??
?
'CCN_1Conv_x0/convB10/kernel/Adam_1/readIdentity"CCN_1Conv_x0/convB10/kernel/Adam_1"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??
?
0CCN_1Conv_x0/convB10/bias/Adam/Initializer/zerosConst"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
dtype0*
valueB?*    
?
CCN_1Conv_x0/convB10/bias/Adam
VariableV2"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
shape:?*
_output_shapes	
:?*
dtype0
?
%CCN_1Conv_x0/convB10/bias/Adam/AssignAssignCCN_1Conv_x0/convB10/bias/Adam0CCN_1Conv_x0/convB10/bias/Adam/Initializer/zeros"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0*
_output_shapes	
:?
?
#CCN_1Conv_x0/convB10/bias/Adam/readIdentityCCN_1Conv_x0/convB10/bias/Adam"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
?
2CCN_1Conv_x0/convB10/bias/Adam_1/Initializer/zerosConst"/device:GPU:1*
dtype0*
valueB?*    *,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
_output_shapes	
:?
?
 CCN_1Conv_x0/convB10/bias/Adam_1
VariableV2"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
dtype0*
shape:?
?
'CCN_1Conv_x0/convB10/bias/Adam_1/AssignAssign CCN_1Conv_x0/convB10/bias/Adam_12CCN_1Conv_x0/convB10/bias/Adam_1/Initializer/zeros"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
_output_shapes	
:?*
T0
?
%CCN_1Conv_x0/convB10/bias/Adam_1/readIdentity CCN_1Conv_x0/convB10/bias/Adam_1"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
?
BCCN_1Conv_x0/convB20/kernel/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
_output_shapes
:*
dtype0*!
valueB"         
?
8CCN_1Conv_x0/convB20/kernel/Adam/Initializer/zeros/ConstConst"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
?
2CCN_1Conv_x0/convB20/kernel/Adam/Initializer/zerosFillBCCN_1Conv_x0/convB20/kernel/Adam/Initializer/zeros/shape_as_tensor8CCN_1Conv_x0/convB20/kernel/Adam/Initializer/zeros/Const"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
 CCN_1Conv_x0/convB20/kernel/Adam
VariableV2"/device:GPU:1*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
shape:??*$
_output_shapes
:??
?
'CCN_1Conv_x0/convB20/kernel/Adam/AssignAssign CCN_1Conv_x0/convB20/kernel/Adam2CCN_1Conv_x0/convB20/kernel/Adam/Initializer/zeros"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??
?
%CCN_1Conv_x0/convB20/kernel/Adam/readIdentity CCN_1Conv_x0/convB20/kernel/Adam"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
DCCN_1Conv_x0/convB20/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
_output_shapes
:*
dtype0*!
valueB"         
?
:CCN_1Conv_x0/convB20/kernel/Adam_1/Initializer/zeros/ConstConst"/device:GPU:1*
dtype0*
valueB
 *    *
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
4CCN_1Conv_x0/convB20/kernel/Adam_1/Initializer/zerosFillDCCN_1Conv_x0/convB20/kernel/Adam_1/Initializer/zeros/shape_as_tensor:CCN_1Conv_x0/convB20/kernel/Adam_1/Initializer/zeros/Const"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??
?
"CCN_1Conv_x0/convB20/kernel/Adam_1
VariableV2"/device:GPU:1*
shape:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??*
dtype0
?
)CCN_1Conv_x0/convB20/kernel/Adam_1/AssignAssign"CCN_1Conv_x0/convB20/kernel/Adam_14CCN_1Conv_x0/convB20/kernel/Adam_1/Initializer/zeros"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
'CCN_1Conv_x0/convB20/kernel/Adam_1/readIdentity"CCN_1Conv_x0/convB20/kernel/Adam_1"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0
?
0CCN_1Conv_x0/convB20/bias/Adam/Initializer/zerosConst"/device:GPU:1*
valueB?*    *,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
dtype0*
_output_shapes	
:?
?
CCN_1Conv_x0/convB20/bias/Adam
VariableV2"/device:GPU:1*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
shape:?*
_output_shapes	
:?
?
%CCN_1Conv_x0/convB20/bias/Adam/AssignAssignCCN_1Conv_x0/convB20/bias/Adam0CCN_1Conv_x0/convB20/bias/Adam/Initializer/zeros"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
T0
?
#CCN_1Conv_x0/convB20/bias/Adam/readIdentityCCN_1Conv_x0/convB20/bias/Adam"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
_output_shapes	
:?*
T0
?
2CCN_1Conv_x0/convB20/bias/Adam_1/Initializer/zerosConst"/device:GPU:1*
valueB?*    *
_output_shapes	
:?*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
 CCN_1Conv_x0/convB20/bias/Adam_1
VariableV2"/device:GPU:1*
shape:?*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
_output_shapes	
:?
?
'CCN_1Conv_x0/convB20/bias/Adam_1/AssignAssign CCN_1Conv_x0/convB20/bias/Adam_12CCN_1Conv_x0/convB20/bias/Adam_1/Initializer/zeros"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
%CCN_1Conv_x0/convB20/bias/Adam_1/readIdentity CCN_1Conv_x0/convB20/bias/Adam_1"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
_output_shapes	
:?
?
BCCN_1Conv_x0/convA11/kernel/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
_output_shapes
:*!
valueB"         *.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
dtype0
?
8CCN_1Conv_x0/convA11/kernel/Adam/Initializer/zeros/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *    *
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
2CCN_1Conv_x0/convA11/kernel/Adam/Initializer/zerosFillBCCN_1Conv_x0/convA11/kernel/Adam/Initializer/zeros/shape_as_tensor8CCN_1Conv_x0/convA11/kernel/Adam/Initializer/zeros/Const"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0*$
_output_shapes
:??
?
 CCN_1Conv_x0/convA11/kernel/Adam
VariableV2"/device:GPU:1*
dtype0*
shape:??*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
'CCN_1Conv_x0/convA11/kernel/Adam/AssignAssign CCN_1Conv_x0/convA11/kernel/Adam2CCN_1Conv_x0/convA11/kernel/Adam/Initializer/zeros"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
%CCN_1Conv_x0/convA11/kernel/Adam/readIdentity CCN_1Conv_x0/convA11/kernel/Adam"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
DCCN_1Conv_x0/convA11/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
_output_shapes
:*!
valueB"         *
dtype0
?
:CCN_1Conv_x0/convA11/kernel/Adam_1/Initializer/zeros/ConstConst"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
4CCN_1Conv_x0/convA11/kernel/Adam_1/Initializer/zerosFillDCCN_1Conv_x0/convA11/kernel/Adam_1/Initializer/zeros/shape_as_tensor:CCN_1Conv_x0/convA11/kernel/Adam_1/Initializer/zeros/Const"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??
?
"CCN_1Conv_x0/convA11/kernel/Adam_1
VariableV2"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
shape:??*$
_output_shapes
:??*
dtype0
?
)CCN_1Conv_x0/convA11/kernel/Adam_1/AssignAssign"CCN_1Conv_x0/convA11/kernel/Adam_14CCN_1Conv_x0/convA11/kernel/Adam_1/Initializer/zeros"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
'CCN_1Conv_x0/convA11/kernel/Adam_1/readIdentity"CCN_1Conv_x0/convA11/kernel/Adam_1"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??
?
0CCN_1Conv_x0/convA11/bias/Adam/Initializer/zerosConst"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?*
valueB?*    *
dtype0
?
CCN_1Conv_x0/convA11/bias/Adam
VariableV2"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
dtype0*
shape:?*
_output_shapes	
:?
?
%CCN_1Conv_x0/convA11/bias/Adam/AssignAssignCCN_1Conv_x0/convA11/bias/Adam0CCN_1Conv_x0/convA11/bias/Adam/Initializer/zeros"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?*
T0
?
#CCN_1Conv_x0/convA11/bias/Adam/readIdentityCCN_1Conv_x0/convA11/bias/Adam"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?*
T0
?
2CCN_1Conv_x0/convA11/bias/Adam_1/Initializer/zerosConst"/device:GPU:1*
_output_shapes	
:?*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
valueB?*    
?
 CCN_1Conv_x0/convA11/bias/Adam_1
VariableV2"/device:GPU:1*
_output_shapes	
:?*
shape:?*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
dtype0
?
'CCN_1Conv_x0/convA11/bias/Adam_1/AssignAssign CCN_1Conv_x0/convA11/bias/Adam_12CCN_1Conv_x0/convA11/bias/Adam_1/Initializer/zeros"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?
?
%CCN_1Conv_x0/convA11/bias/Adam_1/readIdentity CCN_1Conv_x0/convA11/bias/Adam_1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?*
T0
?
BCCN_1Conv_x0/convB11/kernel/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
dtype0*!
valueB"         *.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
_output_shapes
:
?
8CCN_1Conv_x0/convB11/kernel/Adam/Initializer/zeros/ConstConst"/device:GPU:1*
valueB
 *    *
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
_output_shapes
: 
?
2CCN_1Conv_x0/convB11/kernel/Adam/Initializer/zerosFillBCCN_1Conv_x0/convB11/kernel/Adam/Initializer/zeros/shape_as_tensor8CCN_1Conv_x0/convB11/kernel/Adam/Initializer/zeros/Const"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
 CCN_1Conv_x0/convB11/kernel/Adam
VariableV2"/device:GPU:1*
shape:??*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
dtype0
?
'CCN_1Conv_x0/convB11/kernel/Adam/AssignAssign CCN_1Conv_x0/convB11/kernel/Adam2CCN_1Conv_x0/convB11/kernel/Adam/Initializer/zeros"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0*$
_output_shapes
:??
?
%CCN_1Conv_x0/convB11/kernel/Adam/readIdentity CCN_1Conv_x0/convB11/kernel/Adam"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0*$
_output_shapes
:??
?
DCCN_1Conv_x0/convB11/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
_output_shapes
:*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*!
valueB"         *
dtype0
?
:CCN_1Conv_x0/convB11/kernel/Adam_1/Initializer/zeros/ConstConst"/device:GPU:1*
_output_shapes
: *
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
valueB
 *    
?
4CCN_1Conv_x0/convB11/kernel/Adam_1/Initializer/zerosFillDCCN_1Conv_x0/convB11/kernel/Adam_1/Initializer/zeros/shape_as_tensor:CCN_1Conv_x0/convB11/kernel/Adam_1/Initializer/zeros/Const"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
"CCN_1Conv_x0/convB11/kernel/Adam_1
VariableV2"/device:GPU:1*
dtype0*$
_output_shapes
:??*
shape:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
)CCN_1Conv_x0/convB11/kernel/Adam_1/AssignAssign"CCN_1Conv_x0/convB11/kernel/Adam_14CCN_1Conv_x0/convB11/kernel/Adam_1/Initializer/zeros"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*$
_output_shapes
:??*
T0
?
'CCN_1Conv_x0/convB11/kernel/Adam_1/readIdentity"CCN_1Conv_x0/convB11/kernel/Adam_1"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
0CCN_1Conv_x0/convB11/bias/Adam/Initializer/zerosConst"/device:GPU:1*
valueB?*    *
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?
?
CCN_1Conv_x0/convB11/bias/Adam
VariableV2"/device:GPU:1*
shape:?*
_output_shapes	
:?*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
%CCN_1Conv_x0/convB11/bias/Adam/AssignAssignCCN_1Conv_x0/convB11/bias/Adam0CCN_1Conv_x0/convB11/bias/Adam/Initializer/zeros"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
T0
?
#CCN_1Conv_x0/convB11/bias/Adam/readIdentityCCN_1Conv_x0/convB11/bias/Adam"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
2CCN_1Conv_x0/convB11/bias/Adam_1/Initializer/zerosConst"/device:GPU:1*
dtype0*
_output_shapes	
:?*
valueB?*    *,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
 CCN_1Conv_x0/convB11/bias/Adam_1
VariableV2"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?*
shape:?*
dtype0
?
'CCN_1Conv_x0/convB11/bias/Adam_1/AssignAssign CCN_1Conv_x0/convB11/bias/Adam_12CCN_1Conv_x0/convB11/bias/Adam_1/Initializer/zeros"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
T0*
_output_shapes	
:?
?
%CCN_1Conv_x0/convB11/bias/Adam_1/readIdentity CCN_1Conv_x0/convB11/bias/Adam_1"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
BCCN_1Conv_x0/convB21/kernel/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
_output_shapes
:*!
valueB"         *
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
8CCN_1Conv_x0/convB21/kernel/Adam/Initializer/zeros/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *    *
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
2CCN_1Conv_x0/convB21/kernel/Adam/Initializer/zerosFillBCCN_1Conv_x0/convB21/kernel/Adam/Initializer/zeros/shape_as_tensor8CCN_1Conv_x0/convB21/kernel/Adam/Initializer/zeros/Const"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
 CCN_1Conv_x0/convB21/kernel/Adam
VariableV2"/device:GPU:1*$
_output_shapes
:??*
dtype0*
shape:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
'CCN_1Conv_x0/convB21/kernel/Adam/AssignAssign CCN_1Conv_x0/convB21/kernel/Adam2CCN_1Conv_x0/convB21/kernel/Adam/Initializer/zeros"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??
?
%CCN_1Conv_x0/convB21/kernel/Adam/readIdentity CCN_1Conv_x0/convB21/kernel/Adam"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??*
T0
?
DCCN_1Conv_x0/convB21/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
_output_shapes
:*!
valueB"         *.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
dtype0
?
:CCN_1Conv_x0/convB21/kernel/Adam_1/Initializer/zeros/ConstConst"/device:GPU:1*
dtype0*
valueB
 *    *
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
4CCN_1Conv_x0/convB21/kernel/Adam_1/Initializer/zerosFillDCCN_1Conv_x0/convB21/kernel/Adam_1/Initializer/zeros/shape_as_tensor:CCN_1Conv_x0/convB21/kernel/Adam_1/Initializer/zeros/Const"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
T0*$
_output_shapes
:??
?
"CCN_1Conv_x0/convB21/kernel/Adam_1
VariableV2"/device:GPU:1*
shape:??*
dtype0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
)CCN_1Conv_x0/convB21/kernel/Adam_1/AssignAssign"CCN_1Conv_x0/convB21/kernel/Adam_14CCN_1Conv_x0/convB21/kernel/Adam_1/Initializer/zeros"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
T0
?
'CCN_1Conv_x0/convB21/kernel/Adam_1/readIdentity"CCN_1Conv_x0/convB21/kernel/Adam_1"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
0CCN_1Conv_x0/convB21/bias/Adam/Initializer/zerosConst"/device:GPU:1*
valueB?*    *
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
dtype0
?
CCN_1Conv_x0/convB21/bias/Adam
VariableV2"/device:GPU:1*
dtype0*
_output_shapes	
:?*
shape:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
%CCN_1Conv_x0/convB21/bias/Adam/AssignAssignCCN_1Conv_x0/convB21/bias/Adam0CCN_1Conv_x0/convB21/bias/Adam/Initializer/zeros"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0*
_output_shapes	
:?
?
#CCN_1Conv_x0/convB21/bias/Adam/readIdentityCCN_1Conv_x0/convB21/bias/Adam"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes	
:?*
T0
?
2CCN_1Conv_x0/convB21/bias/Adam_1/Initializer/zerosConst"/device:GPU:1*
_output_shapes	
:?*
dtype0*
valueB?*    *,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
 CCN_1Conv_x0/convB21/bias/Adam_1
VariableV2"/device:GPU:1*
shape:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
dtype0*
_output_shapes	
:?
?
'CCN_1Conv_x0/convB21/bias/Adam_1/AssignAssign CCN_1Conv_x0/convB21/bias/Adam_12CCN_1Conv_x0/convB21/bias/Adam_1/Initializer/zeros"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes	
:?
?
%CCN_1Conv_x0/convB21/bias/Adam_1/readIdentity CCN_1Conv_x0/convB21/bias/Adam_1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes	
:?*
T0
?
6Conv_out__/beta/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
dtype0*
_output_shapes
:*
valueB:?
?
,Conv_out__/beta/Adam/Initializer/zeros/ConstConst"/device:GPU:1*
dtype0*"
_class
loc:@Conv_out__/beta*
_output_shapes
: *
valueB
 *    
?
&Conv_out__/beta/Adam/Initializer/zerosFill6Conv_out__/beta/Adam/Initializer/zeros/shape_as_tensor,Conv_out__/beta/Adam/Initializer/zeros/Const"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
T0*
_output_shapes	
:?
?
Conv_out__/beta/Adam
VariableV2"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
dtype0*
_output_shapes	
:?*
shape:?
?
Conv_out__/beta/Adam/AssignAssignConv_out__/beta/Adam&Conv_out__/beta/Adam/Initializer/zeros"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
T0*
_output_shapes	
:?
?
Conv_out__/beta/Adam/readIdentityConv_out__/beta/Adam"/device:GPU:1*
_output_shapes	
:?*
T0*"
_class
loc:@Conv_out__/beta
?
8Conv_out__/beta/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
dtype0*
_output_shapes
:*
valueB:?*"
_class
loc:@Conv_out__/beta
?
.Conv_out__/beta/Adam_1/Initializer/zeros/ConstConst"/device:GPU:1*
_output_shapes
: *"
_class
loc:@Conv_out__/beta*
dtype0*
valueB
 *    
?
(Conv_out__/beta/Adam_1/Initializer/zerosFill8Conv_out__/beta/Adam_1/Initializer/zeros/shape_as_tensor.Conv_out__/beta/Adam_1/Initializer/zeros/Const"/device:GPU:1*
_output_shapes	
:?*
T0*"
_class
loc:@Conv_out__/beta
?
Conv_out__/beta/Adam_1
VariableV2"/device:GPU:1*
dtype0*
_output_shapes	
:?*"
_class
loc:@Conv_out__/beta*
shape:?
?
Conv_out__/beta/Adam_1/AssignAssignConv_out__/beta/Adam_1(Conv_out__/beta/Adam_1/Initializer/zeros"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
_output_shapes	
:?*
T0
?
Conv_out__/beta/Adam_1/readIdentityConv_out__/beta/Adam_1"/device:GPU:1*
_output_shapes	
:?*
T0*"
_class
loc:@Conv_out__/beta
?
7Conv_out__/gamma/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
dtype0*
_output_shapes
:*
valueB:?*#
_class
loc:@Conv_out__/gamma
?
-Conv_out__/gamma/Adam/Initializer/zeros/ConstConst"/device:GPU:1*
valueB
 *    *
dtype0*
_output_shapes
: *#
_class
loc:@Conv_out__/gamma
?
'Conv_out__/gamma/Adam/Initializer/zerosFill7Conv_out__/gamma/Adam/Initializer/zeros/shape_as_tensor-Conv_out__/gamma/Adam/Initializer/zeros/Const"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
T0*
_output_shapes	
:?
?
Conv_out__/gamma/Adam
VariableV2"/device:GPU:1*
shape:?*
_output_shapes	
:?*
dtype0*#
_class
loc:@Conv_out__/gamma
?
Conv_out__/gamma/Adam/AssignAssignConv_out__/gamma/Adam'Conv_out__/gamma/Adam/Initializer/zeros"/device:GPU:1*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma*
T0
?
Conv_out__/gamma/Adam/readIdentityConv_out__/gamma/Adam"/device:GPU:1*
T0*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma
?
9Conv_out__/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
_output_shapes
:*#
_class
loc:@Conv_out__/gamma*
dtype0*
valueB:?
?
/Conv_out__/gamma/Adam_1/Initializer/zeros/ConstConst"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
_output_shapes
: *
valueB
 *    *
dtype0
?
)Conv_out__/gamma/Adam_1/Initializer/zerosFill9Conv_out__/gamma/Adam_1/Initializer/zeros/shape_as_tensor/Conv_out__/gamma/Adam_1/Initializer/zeros/Const"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
T0*
_output_shapes	
:?
?
Conv_out__/gamma/Adam_1
VariableV2"/device:GPU:1*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma*
shape:?*
dtype0
?
Conv_out__/gamma/Adam_1/AssignAssignConv_out__/gamma/Adam_1)Conv_out__/gamma/Adam_1/Initializer/zeros"/device:GPU:1*
T0*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma
?
Conv_out__/gamma/Adam_1/readIdentityConv_out__/gamma/Adam_1"/device:GPU:1*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma*
T0
?
IReconstruction_Output/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
valueB"      *5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:*
dtype0
?
?Reconstruction_Output/dense/kernel/Adam/Initializer/zeros/ConstConst"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
?
9Reconstruction_Output/dense/kernel/Adam/Initializer/zerosFillIReconstruction_Output/dense/kernel/Adam/Initializer/zeros/shape_as_tensor?Reconstruction_Output/dense/kernel/Adam/Initializer/zeros/Const"/device:GPU:1*
_output_shapes
:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
T0
?
'Reconstruction_Output/dense/kernel/Adam
VariableV2"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
shape:	?*
_output_shapes
:	?*
dtype0
?
.Reconstruction_Output/dense/kernel/Adam/AssignAssign'Reconstruction_Output/dense/kernel/Adam9Reconstruction_Output/dense/kernel/Adam/Initializer/zeros"/device:GPU:1*
_output_shapes
:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
T0
?
,Reconstruction_Output/dense/kernel/Adam/readIdentity'Reconstruction_Output/dense/kernel/Adam"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?*
T0
?
KReconstruction_Output/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
_output_shapes
:*
valueB"      *
dtype0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
AReconstruction_Output/dense/kernel/Adam_1/Initializer/zeros/ConstConst"/device:GPU:1*
valueB
 *    *
dtype0*
_output_shapes
: *5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
;Reconstruction_Output/dense/kernel/Adam_1/Initializer/zerosFillKReconstruction_Output/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorAReconstruction_Output/dense/kernel/Adam_1/Initializer/zeros/Const"/device:GPU:1*
_output_shapes
:	?*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
)Reconstruction_Output/dense/kernel/Adam_1
VariableV2"/device:GPU:1*
dtype0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?*
shape:	?
?
0Reconstruction_Output/dense/kernel/Adam_1/AssignAssign)Reconstruction_Output/dense/kernel/Adam_1;Reconstruction_Output/dense/kernel/Adam_1/Initializer/zeros"/device:GPU:1*
T0*
_output_shapes
:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
.Reconstruction_Output/dense/kernel/Adam_1/readIdentity)Reconstruction_Output/dense/kernel/Adam_1"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?*
T0
?
7Reconstruction_Output/dense/bias/Adam/Initializer/zerosConst"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
dtype0*
valueB*    *
_output_shapes
:
?
%Reconstruction_Output/dense/bias/Adam
VariableV2"/device:GPU:1*
shape:*
dtype0*
_output_shapes
:*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
,Reconstruction_Output/dense/bias/Adam/AssignAssign%Reconstruction_Output/dense/bias/Adam7Reconstruction_Output/dense/bias/Adam/Initializer/zeros"/device:GPU:1*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
:
?
*Reconstruction_Output/dense/bias/Adam/readIdentity%Reconstruction_Output/dense/bias/Adam"/device:GPU:1*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
:
?
9Reconstruction_Output/dense/bias/Adam_1/Initializer/zerosConst"/device:GPU:1*
dtype0*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
:*
valueB*    
?
'Reconstruction_Output/dense/bias/Adam_1
VariableV2"/device:GPU:1*
_output_shapes
:*
dtype0*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
shape:
?
.Reconstruction_Output/dense/bias/Adam_1/AssignAssign'Reconstruction_Output/dense/bias/Adam_19Reconstruction_Output/dense/bias/Adam_1/Initializer/zeros"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
:*
T0
?
,Reconstruction_Output/dense/bias/Adam_1/readIdentity'Reconstruction_Output/dense/bias/Adam_1"/device:GPU:1*
_output_shapes
:*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
3dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
dtype0*
valueB"       *
_class
loc:@dense/kernel*
_output_shapes
:
?
)dense/kernel/Adam/Initializer/zeros/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *    *
dtype0*
_class
loc:@dense/kernel
?
#dense/kernel/Adam/Initializer/zerosFill3dense/kernel/Adam/Initializer/zeros/shape_as_tensor)dense/kernel/Adam/Initializer/zeros/Const"/device:GPU:1*
_class
loc:@dense/kernel*
T0*
_output_shapes
:	? 
?
dense/kernel/Adam
VariableV2"/device:GPU:1*
dtype0*
_class
loc:@dense/kernel*
shape:	? *
_output_shapes
:	? 
?
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros"/device:GPU:1*
_output_shapes
:	? *
T0*
_class
loc:@dense/kernel
?
dense/kernel/Adam/readIdentitydense/kernel/Adam"/device:GPU:1*
_output_shapes
:	? *
_class
loc:@dense/kernel*
T0
?
5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
valueB"       *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
?
+dense/kernel/Adam_1/Initializer/zeros/ConstConst"/device:GPU:1*
_class
loc:@dense/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
?
%dense/kernel/Adam_1/Initializer/zerosFill5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor+dense/kernel/Adam_1/Initializer/zeros/Const"/device:GPU:1*
_output_shapes
:	? *
T0*
_class
loc:@dense/kernel
?
dense/kernel/Adam_1
VariableV2"/device:GPU:1*
shape:	? *
dtype0*
_class
loc:@dense/kernel*
_output_shapes
:	? 
?
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros"/device:GPU:1*
T0*
_output_shapes
:	? *
_class
loc:@dense/kernel
?
dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1"/device:GPU:1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	? 
?
!dense/bias/Adam/Initializer/zerosConst"/device:GPU:1*
_class
loc:@dense/bias*
_output_shapes
: *
valueB *    *
dtype0
?
dense/bias/Adam
VariableV2"/device:GPU:1*
_class
loc:@dense/bias*
shape: *
_output_shapes
: *
dtype0
?
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/zeros"/device:GPU:1*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
?
dense/bias/Adam/readIdentitydense/bias/Adam"/device:GPU:1*
T0*
_output_shapes
: *
_class
loc:@dense/bias
?
#dense/bias/Adam_1/Initializer/zerosConst"/device:GPU:1*
valueB *    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
?
dense/bias/Adam_1
VariableV2"/device:GPU:1*
_output_shapes
: *
dtype0*
_class
loc:@dense/bias*
shape: 
?
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros"/device:GPU:1*
T0*
_output_shapes
: *
_class
loc:@dense/bias
?
dense/bias/Adam_1/readIdentitydense/bias/Adam_1"/device:GPU:1*
_class
loc:@dense/bias*
_output_shapes
: *
T0
?
EFCU_muiltDense_x0/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes
:*
valueB"        *
dtype0
?
;FCU_muiltDense_x0/dense/kernel/Adam/Initializer/zeros/ConstConst"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
dtype0*
valueB
 *    *
_output_shapes
: 
?
5FCU_muiltDense_x0/dense/kernel/Adam/Initializer/zerosFillEFCU_muiltDense_x0/dense/kernel/Adam/Initializer/zeros/shape_as_tensor;FCU_muiltDense_x0/dense/kernel/Adam/Initializer/zeros/Const"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
T0*
_output_shapes

:  
?
#FCU_muiltDense_x0/dense/kernel/Adam
VariableV2"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
shape
:  *
dtype0*
_output_shapes

:  
?
*FCU_muiltDense_x0/dense/kernel/Adam/AssignAssign#FCU_muiltDense_x0/dense/kernel/Adam5FCU_muiltDense_x0/dense/kernel/Adam/Initializer/zeros"/device:GPU:1*
_output_shapes

:  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
T0
?
(FCU_muiltDense_x0/dense/kernel/Adam/readIdentity#FCU_muiltDense_x0/dense/kernel/Adam"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes

:  *
T0
?
GFCU_muiltDense_x0/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst"/device:GPU:1*
valueB"        *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
dtype0*
_output_shapes
:
?
=FCU_muiltDense_x0/dense/kernel/Adam_1/Initializer/zeros/ConstConst"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
?
7FCU_muiltDense_x0/dense/kernel/Adam_1/Initializer/zerosFillGFCU_muiltDense_x0/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor=FCU_muiltDense_x0/dense/kernel/Adam_1/Initializer/zeros/Const"/device:GPU:1*
_output_shapes

:  *
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
%FCU_muiltDense_x0/dense/kernel/Adam_1
VariableV2"/device:GPU:1*
shape
:  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
dtype0*
_output_shapes

:  
?
,FCU_muiltDense_x0/dense/kernel/Adam_1/AssignAssign%FCU_muiltDense_x0/dense/kernel/Adam_17FCU_muiltDense_x0/dense/kernel/Adam_1/Initializer/zeros"/device:GPU:1*
T0*
_output_shapes

:  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
*FCU_muiltDense_x0/dense/kernel/Adam_1/readIdentity%FCU_muiltDense_x0/dense/kernel/Adam_1"/device:GPU:1*
_output_shapes

:  *
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
3FCU_muiltDense_x0/dense/bias/Adam/Initializer/zerosConst"/device:GPU:1*
valueB *    */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
dtype0*
_output_shapes
: 
?
!FCU_muiltDense_x0/dense/bias/Adam
VariableV2"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
shape: *
dtype0*
_output_shapes
: 
?
(FCU_muiltDense_x0/dense/bias/Adam/AssignAssign!FCU_muiltDense_x0/dense/bias/Adam3FCU_muiltDense_x0/dense/bias/Adam/Initializer/zeros"/device:GPU:1*
T0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
_output_shapes
: 
?
&FCU_muiltDense_x0/dense/bias/Adam/readIdentity!FCU_muiltDense_x0/dense/bias/Adam"/device:GPU:1*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0
?
5FCU_muiltDense_x0/dense/bias/Adam_1/Initializer/zerosConst"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
_output_shapes
: *
dtype0*
valueB *    
?
#FCU_muiltDense_x0/dense/bias/Adam_1
VariableV2"/device:GPU:1*
dtype0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
shape: *
_output_shapes
: 
?
*FCU_muiltDense_x0/dense/bias/Adam_1/AssignAssign#FCU_muiltDense_x0/dense/bias/Adam_15FCU_muiltDense_x0/dense/bias/Adam_1/Initializer/zeros"/device:GPU:1*
T0*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
(FCU_muiltDense_x0/dense/bias/Adam_1/readIdentity#FCU_muiltDense_x0/dense/bias/Adam_1"/device:GPU:1*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0
?
-FCU_muiltDense_x0/beta/Adam/Initializer/zerosConst"/device:GPU:1*
_output_shapes
: *)
_class
loc:@FCU_muiltDense_x0/beta*
dtype0*
valueB *    
?
FCU_muiltDense_x0/beta/Adam
VariableV2"/device:GPU:1*
_output_shapes
: *)
_class
loc:@FCU_muiltDense_x0/beta*
shape: *
dtype0
?
"FCU_muiltDense_x0/beta/Adam/AssignAssignFCU_muiltDense_x0/beta/Adam-FCU_muiltDense_x0/beta/Adam/Initializer/zeros"/device:GPU:1*
_output_shapes
: *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
 FCU_muiltDense_x0/beta/Adam/readIdentityFCU_muiltDense_x0/beta/Adam"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta*
T0*
_output_shapes
: 
?
/FCU_muiltDense_x0/beta/Adam_1/Initializer/zerosConst"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta*
valueB *    *
_output_shapes
: *
dtype0
?
FCU_muiltDense_x0/beta/Adam_1
VariableV2"/device:GPU:1*
shape: *
_output_shapes
: *
dtype0*)
_class
loc:@FCU_muiltDense_x0/beta
?
$FCU_muiltDense_x0/beta/Adam_1/AssignAssignFCU_muiltDense_x0/beta/Adam_1/FCU_muiltDense_x0/beta/Adam_1/Initializer/zeros"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta*
_output_shapes
: *
T0
?
"FCU_muiltDense_x0/beta/Adam_1/readIdentityFCU_muiltDense_x0/beta/Adam_1"/device:GPU:1*
_output_shapes
: *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
.FCU_muiltDense_x0/gamma/Adam/Initializer/zerosConst"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
valueB *    *
dtype0*
_output_shapes
: 
?
FCU_muiltDense_x0/gamma/Adam
VariableV2"/device:GPU:1*
shape: **
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: *
dtype0
?
#FCU_muiltDense_x0/gamma/Adam/AssignAssignFCU_muiltDense_x0/gamma/Adam.FCU_muiltDense_x0/gamma/Adam/Initializer/zeros"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
T0*
_output_shapes
: 
?
!FCU_muiltDense_x0/gamma/Adam/readIdentityFCU_muiltDense_x0/gamma/Adam"/device:GPU:1*
T0*
_output_shapes
: **
_class 
loc:@FCU_muiltDense_x0/gamma
?
0FCU_muiltDense_x0/gamma/Adam_1/Initializer/zerosConst"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
dtype0*
_output_shapes
: *
valueB *    
?
FCU_muiltDense_x0/gamma/Adam_1
VariableV2"/device:GPU:1*
_output_shapes
: *
dtype0*
shape: **
_class 
loc:@FCU_muiltDense_x0/gamma
?
%FCU_muiltDense_x0/gamma/Adam_1/AssignAssignFCU_muiltDense_x0/gamma/Adam_10FCU_muiltDense_x0/gamma/Adam_1/Initializer/zeros"/device:GPU:1*
T0**
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: 
?
#FCU_muiltDense_x0/gamma/Adam_1/readIdentityFCU_muiltDense_x0/gamma/Adam_1"/device:GPU:1*
T0**
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: 
?
+Output_/dense/kernel/Adam/Initializer/zerosConst"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*
_output_shapes

: *
dtype0*
valueB *    
?
Output_/dense/kernel/Adam
VariableV2"/device:GPU:1*
shape
: *'
_class
loc:@Output_/dense/kernel*
dtype0*
_output_shapes

: 
?
 Output_/dense/kernel/Adam/AssignAssignOutput_/dense/kernel/Adam+Output_/dense/kernel/Adam/Initializer/zeros"/device:GPU:1*
_output_shapes

: *
T0*'
_class
loc:@Output_/dense/kernel
?
Output_/dense/kernel/Adam/readIdentityOutput_/dense/kernel/Adam"/device:GPU:1*
_output_shapes

: *'
_class
loc:@Output_/dense/kernel*
T0
?
-Output_/dense/kernel/Adam_1/Initializer/zerosConst"/device:GPU:1*
valueB *    *'
_class
loc:@Output_/dense/kernel*
dtype0*
_output_shapes

: 
?
Output_/dense/kernel/Adam_1
VariableV2"/device:GPU:1*
dtype0*
_output_shapes

: *
shape
: *'
_class
loc:@Output_/dense/kernel
?
"Output_/dense/kernel/Adam_1/AssignAssignOutput_/dense/kernel/Adam_1-Output_/dense/kernel/Adam_1/Initializer/zeros"/device:GPU:1*
T0*'
_class
loc:@Output_/dense/kernel*
_output_shapes

: 
?
 Output_/dense/kernel/Adam_1/readIdentityOutput_/dense/kernel/Adam_1"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*
T0*
_output_shapes

: 
?
)Output_/dense/bias/Adam/Initializer/zerosConst"/device:GPU:1*
valueB*    *
_output_shapes
:*%
_class
loc:@Output_/dense/bias*
dtype0
?
Output_/dense/bias/Adam
VariableV2"/device:GPU:1*
shape:*
dtype0*
_output_shapes
:*%
_class
loc:@Output_/dense/bias
?
Output_/dense/bias/Adam/AssignAssignOutput_/dense/bias/Adam)Output_/dense/bias/Adam/Initializer/zeros"/device:GPU:1*
_output_shapes
:*
T0*%
_class
loc:@Output_/dense/bias
?
Output_/dense/bias/Adam/readIdentityOutput_/dense/bias/Adam"/device:GPU:1*
_output_shapes
:*
T0*%
_class
loc:@Output_/dense/bias
?
+Output_/dense/bias/Adam_1/Initializer/zerosConst"/device:GPU:1*
dtype0*%
_class
loc:@Output_/dense/bias*
valueB*    *
_output_shapes
:
?
Output_/dense/bias/Adam_1
VariableV2"/device:GPU:1*
dtype0*
shape:*%
_class
loc:@Output_/dense/bias*
_output_shapes
:
?
 Output_/dense/bias/Adam_1/AssignAssignOutput_/dense/bias/Adam_1+Output_/dense/bias/Adam_1/Initializer/zeros"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
T0*
_output_shapes
:
?
Output_/dense/bias/Adam_1/readIdentityOutput_/dense/bias/Adam_1"/device:GPU:1*
_output_shapes
:*
T0*%
_class
loc:@Output_/dense/bias
^

Adam/beta1Const"/device:CPU:0*
dtype0*
valueB
 *fff?*
_output_shapes
: 
^

Adam/beta2Const"/device:CPU:0*
dtype0*
valueB
 *w??*
_output_shapes
: 
`
Adam/epsilonConst"/device:CPU:0*
valueB
 *w?+2*
_output_shapes
: *
dtype0
?
1Adam/update_CCN_1Conv_x0/convA10/kernel/ApplyAdam	ApplyAdamCCN_1Conv_x0/convA10/kernel CCN_1Conv_x0/convA10/kernel/Adam"CCN_1Conv_x0/convA10/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0*#
_output_shapes
:?
?
/Adam/update_CCN_1Conv_x0/convA10/bias/ApplyAdam	ApplyAdamCCN_1Conv_x0/convA10/biasCCN_1Conv_x0/convA10/bias/Adam CCN_1Conv_x0/convA10/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?*
T0
?
1Adam/update_CCN_1Conv_x0/convB10/kernel/ApplyAdam	ApplyAdamCCN_1Conv_x0/convB10/kernel CCN_1Conv_x0/convB10/kernel/Adam"CCN_1Conv_x0/convB10/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_2"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??*
T0
?
/Adam/update_CCN_1Conv_x0/convB10/bias/ApplyAdam	ApplyAdamCCN_1Conv_x0/convB10/biasCCN_1Conv_x0/convB10/bias/Adam CCN_1Conv_x0/convB10/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_3"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0*
_output_shapes	
:?
?
1Adam/update_CCN_1Conv_x0/convB20/kernel/ApplyAdam	ApplyAdamCCN_1Conv_x0/convB20/kernel CCN_1Conv_x0/convB20/kernel/Adam"CCN_1Conv_x0/convB20/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_4"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
/Adam/update_CCN_1Conv_x0/convB20/bias/ApplyAdam	ApplyAdamCCN_1Conv_x0/convB20/biasCCN_1Conv_x0/convB20/bias/Adam CCN_1Conv_x0/convB20/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_5"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
1Adam/update_CCN_1Conv_x0/convA11/kernel/ApplyAdam	ApplyAdamCCN_1Conv_x0/convA11/kernel CCN_1Conv_x0/convA11/kernel/Adam"CCN_1Conv_x0/convA11/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_6"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0*$
_output_shapes
:??
?
/Adam/update_CCN_1Conv_x0/convA11/bias/ApplyAdam	ApplyAdamCCN_1Conv_x0/convA11/biasCCN_1Conv_x0/convA11/bias/Adam CCN_1Conv_x0/convA11/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_7"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0
?
1Adam/update_CCN_1Conv_x0/convB11/kernel/ApplyAdam	ApplyAdamCCN_1Conv_x0/convB11/kernel CCN_1Conv_x0/convB11/kernel/Adam"CCN_1Conv_x0/convB11/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_8"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0
?
/Adam/update_CCN_1Conv_x0/convB11/bias/ApplyAdam	ApplyAdamCCN_1Conv_x0/convB11/biasCCN_1Conv_x0/convB11/bias/Adam CCN_1Conv_x0/convB11/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_9"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?*
T0
?
1Adam/update_CCN_1Conv_x0/convB21/kernel/ApplyAdam	ApplyAdamCCN_1Conv_x0/convB21/kernel CCN_1Conv_x0/convB21/kernel/Adam"CCN_1Conv_x0/convB21/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_10"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??*
T0
?
/Adam/update_CCN_1Conv_x0/convB21/bias/ApplyAdam	ApplyAdamCCN_1Conv_x0/convB21/biasCCN_1Conv_x0/convB21/bias/Adam CCN_1Conv_x0/convB21/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_11"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes	
:?
?
%Adam/update_Conv_out__/beta/ApplyAdam	ApplyAdamConv_out__/betaConv_out__/beta/AdamConv_out__/beta/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_12"/device:GPU:1*
_output_shapes	
:?*"
_class
loc:@Conv_out__/beta*
T0
?
&Adam/update_Conv_out__/gamma/ApplyAdam	ApplyAdamConv_out__/gammaConv_out__/gamma/AdamConv_out__/gamma/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_13"/device:GPU:1*
T0*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma
?
8Adam/update_Reconstruction_Output/dense/kernel/ApplyAdam	ApplyAdam"Reconstruction_Output/dense/kernel'Reconstruction_Output/dense/kernel/Adam)Reconstruction_Output/dense/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_14"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?*
T0
?
6Adam/update_Reconstruction_Output/dense/bias/ApplyAdam	ApplyAdam Reconstruction_Output/dense/bias%Reconstruction_Output/dense/bias/Adam'Reconstruction_Output/dense/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_15"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
:*
T0
?
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_16"/device:GPU:1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	? 
?
 Adam/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_17"/device:GPU:1*
_output_shapes
: *
T0*
_class
loc:@dense/bias
?
4Adam/update_FCU_muiltDense_x0/dense/kernel/ApplyAdam	ApplyAdamFCU_muiltDense_x0/dense/kernel#FCU_muiltDense_x0/dense/kernel/Adam%FCU_muiltDense_x0/dense/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_18"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes

:  *
T0
?
2Adam/update_FCU_muiltDense_x0/dense/bias/ApplyAdam	ApplyAdamFCU_muiltDense_x0/dense/bias!FCU_muiltDense_x0/dense/bias/Adam#FCU_muiltDense_x0/dense/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_19"/device:GPU:1*
T0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
_output_shapes
: 
?
,Adam/update_FCU_muiltDense_x0/beta/ApplyAdam	ApplyAdamFCU_muiltDense_x0/betaFCU_muiltDense_x0/beta/AdamFCU_muiltDense_x0/beta/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_20"/device:GPU:1*
T0*
_output_shapes
: *)
_class
loc:@FCU_muiltDense_x0/beta
?
-Adam/update_FCU_muiltDense_x0/gamma/ApplyAdam	ApplyAdamFCU_muiltDense_x0/gammaFCU_muiltDense_x0/gamma/AdamFCU_muiltDense_x0/gamma/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_21"/device:GPU:1*
T0**
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: 
?
*Adam/update_Output_/dense/kernel/ApplyAdam	ApplyAdamOutput_/dense/kernelOutput_/dense/kernel/AdamOutput_/dense/kernel/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_22"/device:GPU:1*
_output_shapes

: *
T0*'
_class
loc:@Output_/dense/kernel
?
(Adam/update_Output_/dense/bias/ApplyAdam	ApplyAdamOutput_/dense/biasOutput_/dense/bias/AdamOutput_/dense/bias/Adam_1beta1_power/readbeta2_power/readExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilonMean_23"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
T0*
_output_shapes
:
?

Adam/mulMulbeta1_power/read
Adam/beta10^Adam/update_CCN_1Conv_x0/convA10/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convA10/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convA11/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convA11/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB10/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB10/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB11/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB11/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB20/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB20/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB21/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB21/kernel/ApplyAdam&^Adam/update_Conv_out__/beta/ApplyAdam'^Adam/update_Conv_out__/gamma/ApplyAdam-^Adam/update_FCU_muiltDense_x0/beta/ApplyAdam3^Adam/update_FCU_muiltDense_x0/dense/bias/ApplyAdam5^Adam/update_FCU_muiltDense_x0/dense/kernel/ApplyAdam.^Adam/update_FCU_muiltDense_x0/gamma/ApplyAdam)^Adam/update_Output_/dense/bias/ApplyAdam+^Adam/update_Output_/dense/kernel/ApplyAdam7^Adam/update_Reconstruction_Output/dense/bias/ApplyAdam9^Adam/update_Reconstruction_Output/dense/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes
: 
?
Adam/AssignAssignbeta1_powerAdam/mul"/device:GPU:1*
use_locking( *
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes
: 
?


Adam/mul_1Mulbeta2_power/read
Adam/beta20^Adam/update_CCN_1Conv_x0/convA10/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convA10/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convA11/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convA11/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB10/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB10/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB11/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB11/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB20/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB20/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB21/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB21/kernel/ApplyAdam&^Adam/update_Conv_out__/beta/ApplyAdam'^Adam/update_Conv_out__/gamma/ApplyAdam-^Adam/update_FCU_muiltDense_x0/beta/ApplyAdam3^Adam/update_FCU_muiltDense_x0/dense/bias/ApplyAdam5^Adam/update_FCU_muiltDense_x0/dense/kernel/ApplyAdam.^Adam/update_FCU_muiltDense_x0/gamma/ApplyAdam)^Adam/update_Output_/dense/bias/ApplyAdam+^Adam/update_Output_/dense/kernel/ApplyAdam7^Adam/update_Reconstruction_Output/dense/bias/ApplyAdam9^Adam/update_Reconstruction_Output/dense/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam"/device:GPU:1*
_output_shapes
: *
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes
: *
use_locking( 
?	
Adam/updateNoOp^Adam/Assign^Adam/Assign_10^Adam/update_CCN_1Conv_x0/convA10/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convA10/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convA11/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convA11/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB10/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB10/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB11/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB11/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB20/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB20/kernel/ApplyAdam0^Adam/update_CCN_1Conv_x0/convB21/bias/ApplyAdam2^Adam/update_CCN_1Conv_x0/convB21/kernel/ApplyAdam&^Adam/update_Conv_out__/beta/ApplyAdam'^Adam/update_Conv_out__/gamma/ApplyAdam-^Adam/update_FCU_muiltDense_x0/beta/ApplyAdam3^Adam/update_FCU_muiltDense_x0/dense/bias/ApplyAdam5^Adam/update_FCU_muiltDense_x0/dense/kernel/ApplyAdam.^Adam/update_FCU_muiltDense_x0/gamma/ApplyAdam)^Adam/update_Output_/dense/bias/ApplyAdam+^Adam/update_Output_/dense/kernel/ApplyAdam7^Adam/update_Reconstruction_Output/dense/bias/ApplyAdam9^Adam/update_Reconstruction_Output/dense/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam"/device:GPU:1
?

Adam/valueConst^Adam/update"/device:CPU:0*
valueB
 *  ??*
_class
loc:@global_step*
_output_shapes
: *
dtype0
z
Adam	AssignAddglobal_step
Adam/value"/device:CPU:0*
_output_shapes
: *
T0*
_class
loc:@global_step
?
!CCN_1Conv_x0/convA10/kernel_1/tagConst"/device:CPU:0*.
value%B# BCCN_1Conv_x0/convA10/kernel_1*
_output_shapes
: *
dtype0
?
CCN_1Conv_x0/convA10/kernel_1HistogramSummary!CCN_1Conv_x0/convA10/kernel_1/tag CCN_1Conv_x0/convA10/kernel/read"/device:CPU:0*
_output_shapes
: 
?
CCN_1Conv_x0/convA10/bias_1/tagConst"/device:CPU:0*
_output_shapes
: *
dtype0*,
value#B! BCCN_1Conv_x0/convA10/bias_1
?
CCN_1Conv_x0/convA10/bias_1HistogramSummaryCCN_1Conv_x0/convA10/bias_1/tagCCN_1Conv_x0/convA10/bias/read"/device:CPU:0*
_output_shapes
: 
?
!CCN_1Conv_x0/convB10/kernel_1/tagConst"/device:CPU:0*
dtype0*.
value%B# BCCN_1Conv_x0/convB10/kernel_1*
_output_shapes
: 
?
CCN_1Conv_x0/convB10/kernel_1HistogramSummary!CCN_1Conv_x0/convB10/kernel_1/tag CCN_1Conv_x0/convB10/kernel/read"/device:CPU:0*
_output_shapes
: 
?
CCN_1Conv_x0/convB10/bias_1/tagConst"/device:CPU:0*
_output_shapes
: *
dtype0*,
value#B! BCCN_1Conv_x0/convB10/bias_1
?
CCN_1Conv_x0/convB10/bias_1HistogramSummaryCCN_1Conv_x0/convB10/bias_1/tagCCN_1Conv_x0/convB10/bias/read"/device:CPU:0*
_output_shapes
: 
?
!CCN_1Conv_x0/convB20/kernel_1/tagConst"/device:CPU:0*
dtype0*
_output_shapes
: *.
value%B# BCCN_1Conv_x0/convB20/kernel_1
?
CCN_1Conv_x0/convB20/kernel_1HistogramSummary!CCN_1Conv_x0/convB20/kernel_1/tag CCN_1Conv_x0/convB20/kernel/read"/device:CPU:0*
_output_shapes
: 
?
CCN_1Conv_x0/convB20/bias_1/tagConst"/device:CPU:0*
_output_shapes
: *
dtype0*,
value#B! BCCN_1Conv_x0/convB20/bias_1
?
CCN_1Conv_x0/convB20/bias_1HistogramSummaryCCN_1Conv_x0/convB20/bias_1/tagCCN_1Conv_x0/convB20/bias/read"/device:CPU:0*
_output_shapes
: 
?
!CCN_1Conv_x0/convA11/kernel_1/tagConst"/device:CPU:0*
_output_shapes
: *.
value%B# BCCN_1Conv_x0/convA11/kernel_1*
dtype0
?
CCN_1Conv_x0/convA11/kernel_1HistogramSummary!CCN_1Conv_x0/convA11/kernel_1/tag CCN_1Conv_x0/convA11/kernel/read"/device:CPU:0*
_output_shapes
: 
?
CCN_1Conv_x0/convA11/bias_1/tagConst"/device:CPU:0*,
value#B! BCCN_1Conv_x0/convA11/bias_1*
_output_shapes
: *
dtype0
?
CCN_1Conv_x0/convA11/bias_1HistogramSummaryCCN_1Conv_x0/convA11/bias_1/tagCCN_1Conv_x0/convA11/bias/read"/device:CPU:0*
_output_shapes
: 
?
!CCN_1Conv_x0/convB11/kernel_1/tagConst"/device:CPU:0*.
value%B# BCCN_1Conv_x0/convB11/kernel_1*
_output_shapes
: *
dtype0
?
CCN_1Conv_x0/convB11/kernel_1HistogramSummary!CCN_1Conv_x0/convB11/kernel_1/tag CCN_1Conv_x0/convB11/kernel/read"/device:CPU:0*
_output_shapes
: 
?
CCN_1Conv_x0/convB11/bias_1/tagConst"/device:CPU:0*
_output_shapes
: *
dtype0*,
value#B! BCCN_1Conv_x0/convB11/bias_1
?
CCN_1Conv_x0/convB11/bias_1HistogramSummaryCCN_1Conv_x0/convB11/bias_1/tagCCN_1Conv_x0/convB11/bias/read"/device:CPU:0*
_output_shapes
: 
?
!CCN_1Conv_x0/convB21/kernel_1/tagConst"/device:CPU:0*.
value%B# BCCN_1Conv_x0/convB21/kernel_1*
_output_shapes
: *
dtype0
?
CCN_1Conv_x0/convB21/kernel_1HistogramSummary!CCN_1Conv_x0/convB21/kernel_1/tag CCN_1Conv_x0/convB21/kernel/read"/device:CPU:0*
_output_shapes
: 
?
CCN_1Conv_x0/convB21/bias_1/tagConst"/device:CPU:0*
_output_shapes
: *,
value#B! BCCN_1Conv_x0/convB21/bias_1*
dtype0
?
CCN_1Conv_x0/convB21/bias_1HistogramSummaryCCN_1Conv_x0/convB21/bias_1/tagCCN_1Conv_x0/convB21/bias/read"/device:CPU:0*
_output_shapes
: 
v
Conv_out__/beta_1/tagConst"/device:CPU:0*
_output_shapes
: *"
valueB BConv_out__/beta_1*
dtype0
y
Conv_out__/beta_1HistogramSummaryConv_out__/beta_1/tagConv_out__/beta/read"/device:CPU:0*
_output_shapes
: 
x
Conv_out__/gamma_1/tagConst"/device:CPU:0*
dtype0*
_output_shapes
: *#
valueB BConv_out__/gamma_1
|
Conv_out__/gamma_1HistogramSummaryConv_out__/gamma_1/tagConv_out__/gamma/read"/device:CPU:0*
_output_shapes
: 
?
(Reconstruction_Output/dense/kernel_1/tagConst"/device:CPU:0*
dtype0*5
value,B* B$Reconstruction_Output/dense/kernel_1*
_output_shapes
: 
?
$Reconstruction_Output/dense/kernel_1HistogramSummary(Reconstruction_Output/dense/kernel_1/tag'Reconstruction_Output/dense/kernel/read"/device:CPU:0*
_output_shapes
: 
?
&Reconstruction_Output/dense/bias_1/tagConst"/device:CPU:0*3
value*B( B"Reconstruction_Output/dense/bias_1*
_output_shapes
: *
dtype0
?
"Reconstruction_Output/dense/bias_1HistogramSummary&Reconstruction_Output/dense/bias_1/tag%Reconstruction_Output/dense/bias/read"/device:CPU:0*
_output_shapes
: 
p
dense/kernel_1/tagConst"/device:CPU:0*
valueB Bdense/kernel_1*
dtype0*
_output_shapes
: 
p
dense/kernel_1HistogramSummarydense/kernel_1/tagdense/kernel/read"/device:CPU:0*
_output_shapes
: 
l
dense/bias_1/tagConst"/device:CPU:0*
dtype0*
valueB Bdense/bias_1*
_output_shapes
: 
j
dense/bias_1HistogramSummarydense/bias_1/tagdense/bias/read"/device:CPU:0*
_output_shapes
: 
?
$FCU_muiltDense_x0/dense/kernel_1/tagConst"/device:CPU:0*
_output_shapes
: *
dtype0*1
value(B& B FCU_muiltDense_x0/dense/kernel_1
?
 FCU_muiltDense_x0/dense/kernel_1HistogramSummary$FCU_muiltDense_x0/dense/kernel_1/tag#FCU_muiltDense_x0/dense/kernel/read"/device:CPU:0*
_output_shapes
: 
?
"FCU_muiltDense_x0/dense/bias_1/tagConst"/device:CPU:0*/
value&B$ BFCU_muiltDense_x0/dense/bias_1*
dtype0*
_output_shapes
: 
?
FCU_muiltDense_x0/dense/bias_1HistogramSummary"FCU_muiltDense_x0/dense/bias_1/tag!FCU_muiltDense_x0/dense/bias/read"/device:CPU:0*
_output_shapes
: 
?
FCU_muiltDense_x0/beta_1/tagConst"/device:CPU:0*
_output_shapes
: *)
value B BFCU_muiltDense_x0/beta_1*
dtype0
?
FCU_muiltDense_x0/beta_1HistogramSummaryFCU_muiltDense_x0/beta_1/tagFCU_muiltDense_x0/beta/read"/device:CPU:0*
_output_shapes
: 
?
FCU_muiltDense_x0/gamma_1/tagConst"/device:CPU:0*
dtype0**
value!B BFCU_muiltDense_x0/gamma_1*
_output_shapes
: 
?
FCU_muiltDense_x0/gamma_1HistogramSummaryFCU_muiltDense_x0/gamma_1/tagFCU_muiltDense_x0/gamma/read"/device:CPU:0*
_output_shapes
: 
?
Output_/dense/kernel_1/tagConst"/device:CPU:0*
_output_shapes
: *
dtype0*'
valueB BOutput_/dense/kernel_1
?
Output_/dense/kernel_1HistogramSummaryOutput_/dense/kernel_1/tagOutput_/dense/kernel/read"/device:CPU:0*
_output_shapes
: 
|
Output_/dense/bias_1/tagConst"/device:CPU:0*
dtype0*
_output_shapes
: *%
valueB BOutput_/dense/bias_1
?
Output_/dense/bias_1HistogramSummaryOutput_/dense/bias_1/tagOutput_/dense/bias/read"/device:CPU:0*
_output_shapes
: 
x
Outputdata_IdentityMy_GPU_1/Output_/dense/BiasAdd"/device:CPU:0*'
_output_shapes
:?????????*
T0
?
IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convA10/kernel"/device:GPU:1*
_output_shapes
: *
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
u
cond/SwitchSwitchIsVariableInitializedIsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: : 
X
cond/switch_tIdentitycond/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
V
cond/switch_fIdentitycond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
_
cond/pred_idIdentityIsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

f
	cond/readIdentitycond/read/Switch:1"/device:CPU:0*#
_output_shapes
:?*
T0
?
cond/read/Switch	RefSwitchCCN_1Conv_x0/convA10/kernelcond/pred_id"/device:GPU:1*2
_output_shapes 
:?:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0
?
cond/Switch_1Switch6CCN_1Conv_x0/convA10/kernel/Initializer/random_uniformcond/pred_id*
T0*2
_output_shapes 
:?:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
u

cond/MergeMergecond/Switch_1	cond/read"/device:CPU:0*
N*%
_output_shapes
:?: *
T0
?
4CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage
VariableV2"/device:GPU:1*
shape:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?*
dtype0
?
JCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convA10/kernel"/device:GPU:1*
dtype0*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
@CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/SwitchSwitchJCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/IsVariableInitializedJCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: : *
T0
*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
BCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/switch_tIdentityBCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0

?
BCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/switch_fIdentity@CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/Switch"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0
*
_output_shapes
: 
?
ACCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/pred_idIdentityJCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: *
T0
*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
>CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/readIdentityGCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?*
T0
?
ECCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convA10/kernelACCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*
T0*2
_output_shapes 
:?:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
BCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/Switch_1Switch6CCN_1Conv_x0/convA10/kernel/Initializer/random_uniformACCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/pred_id*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*2
_output_shapes 
:?:?*
T0
?
?CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/MergeMergeBCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/Switch_1>CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/read"/device:GPU:1*%
_output_shapes
:?: *
N*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0
?
Econd/read/Switch_CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverageSwitch?CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/Mergecond/pred_id"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*2
_output_shapes 
:?:?
?
>cond/read_CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverageIdentityGcond/read/Switch_CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage:1"/device:GPU:1*#
_output_shapes
:?*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
?cond/Merge_CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverageMergecond/Switch_1>cond/read_CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage"/device:GPU:1*
T0*%
_output_shapes
:?: *
N*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
;CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/AssignAssign4CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage?cond/Merge_CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage"/device:GPU:1*
T0*#
_output_shapes
:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
9CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/readIdentity4CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage"/device:GPU:1*#
_output_shapes
:?*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
IsVariableInitialized_1IsVariableInitializedCCN_1Conv_x0/convA10/bias"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
dtype0*
_output_shapes
: 
{
cond_1/SwitchSwitchIsVariableInitialized_1IsVariableInitialized_1"/device:CPU:0*
T0
*
_output_shapes
: : 
\
cond_1/switch_tIdentitycond_1/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
Z
cond_1/switch_fIdentitycond_1/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
c
cond_1/pred_idIdentityIsVariableInitialized_1"/device:CPU:0*
_output_shapes
: *
T0

b
cond_1/readIdentitycond_1/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
cond_1/read/Switch	RefSwitchCCN_1Conv_x0/convA10/biascond_1/pred_id"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*"
_output_shapes
:?:?
?
cond_1/Switch_1Switch+CCN_1Conv_x0/convA10/bias/Initializer/zeroscond_1/pred_id*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*"
_output_shapes
:?:?
s
cond_1/MergeMergecond_1/Switch_1cond_1/read"/device:CPU:0*
N*
T0*
_output_shapes
	:?: 
?
2CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage
VariableV2"/device:GPU:1*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?*
shape:?
?
HCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convA10/bias"/device:GPU:1*
_output_shapes
: *
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
>CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/SwitchSwitchHCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/IsVariableInitializedHCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes
: : 
?
@CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/switch_tIdentity@CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0
*
_output_shapes
: 
?
@CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/switch_fIdentity>CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/Switch"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes
: *
T0

?
?CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/pred_idIdentityHCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0
*
_output_shapes
: 
?
<CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/readIdentityECCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
CCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convA10/bias?CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0
?
@CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/Switch_1Switch+CCN_1Conv_x0/convA10/bias/Initializer/zeros?CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/pred_id*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*"
_output_shapes
:?:?*
T0
?
=CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/MergeMerge@CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/Switch_1<CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/read"/device:GPU:1*
_output_shapes
	:?: *,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
N*
T0
?
Econd_1/read/Switch_CCN_1Conv_x0/convA10/bias/ExponentialMovingAverageSwitch=CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/Mergecond_1/pred_id"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*"
_output_shapes
:?:?
?
>cond_1/read_CCN_1Conv_x0/convA10/bias/ExponentialMovingAverageIdentityGcond_1/read/Switch_CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage:1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?*
T0
?
?cond_1/Merge_CCN_1Conv_x0/convA10/bias/ExponentialMovingAverageMergecond_1/Switch_1>cond_1/read_CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage"/device:GPU:1*
_output_shapes
	:?: *
T0*
N*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
9CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/AssignAssign2CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage?cond_1/Merge_CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?
?
7CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/readIdentity2CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
IsVariableInitialized_2IsVariableInitializedCCN_1Conv_x0/convB10/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
dtype0*
_output_shapes
: 
{
cond_2/SwitchSwitchIsVariableInitialized_2IsVariableInitialized_2"/device:CPU:0*
T0
*
_output_shapes
: : 
\
cond_2/switch_tIdentitycond_2/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

Z
cond_2/switch_fIdentitycond_2/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
c
cond_2/pred_idIdentityIsVariableInitialized_2"/device:CPU:0*
T0
*
_output_shapes
: 
k
cond_2/readIdentitycond_2/read/Switch:1"/device:CPU:0*$
_output_shapes
:??*
T0
?
cond_2/read/Switch	RefSwitchCCN_1Conv_x0/convB10/kernelcond_2/pred_id"/device:GPU:1*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0
?
cond_2/Switch_1Switch6CCN_1Conv_x0/convB10/kernel/Initializer/random_uniformcond_2/pred_id*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0*4
_output_shapes"
 :??:??
|
cond_2/MergeMergecond_2/Switch_1cond_2/read"/device:CPU:0*
T0*&
_output_shapes
:??: *
N
?
4CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage
VariableV2"/device:GPU:1*
shape:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??*
dtype0
?
JCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB10/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
dtype0*
_output_shapes
: 
?
@CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/SwitchSwitchJCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/IsVariableInitializedJCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
_output_shapes
: : *
T0

?
BCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/switch_tIdentityBCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
T0
*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
_output_shapes
: 
?
BCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/switch_fIdentity@CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/Switch"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0
*
_output_shapes
: 
?
ACCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/pred_idIdentityJCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0
*
_output_shapes
: 
?
>CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/readIdentityGCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??*
T0
?
ECCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB10/kernelACCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0*4
_output_shapes"
 :??:??
?
BCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/Switch_1Switch6CCN_1Conv_x0/convB10/kernel/Initializer/random_uniformACCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/pred_id*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*4
_output_shapes"
 :??:??*
T0
?
?CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/MergeMergeBCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/Switch_1>CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/read"/device:GPU:1*&
_output_shapes
:??: *
T0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
N
?
Gcond_2/read/Switch_CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverageSwitch?CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/Mergecond_2/pred_id"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0*4
_output_shapes"
 :??:??
?
@cond_2/read_CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverageIdentityIcond_2/read/Switch_CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage:1"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0*$
_output_shapes
:??
?
Acond_2/Merge_CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverageMergecond_2/Switch_1@cond_2/read_CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*&
_output_shapes
:??: *
N
?
;CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/AssignAssign4CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverageAcond_2/Merge_CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??*
T0
?
9CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/readIdentity4CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0*$
_output_shapes
:??
?
IsVariableInitialized_3IsVariableInitializedCCN_1Conv_x0/convB10/bias"/device:GPU:1*
_output_shapes
: *
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
{
cond_3/SwitchSwitchIsVariableInitialized_3IsVariableInitialized_3"/device:CPU:0*
T0
*
_output_shapes
: : 
\
cond_3/switch_tIdentitycond_3/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
Z
cond_3/switch_fIdentitycond_3/Switch"/device:CPU:0*
_output_shapes
: *
T0

c
cond_3/pred_idIdentityIsVariableInitialized_3"/device:CPU:0*
_output_shapes
: *
T0

b
cond_3/readIdentitycond_3/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
cond_3/read/Switch	RefSwitchCCN_1Conv_x0/convB10/biascond_3/pred_id"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*"
_output_shapes
:?:?
?
cond_3/Switch_1Switch+CCN_1Conv_x0/convB10/bias/Initializer/zeroscond_3/pred_id*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*"
_output_shapes
:?:?*
T0
s
cond_3/MergeMergecond_3/Switch_1cond_3/read"/device:CPU:0*
T0*
_output_shapes
	:?: *
N
?
2CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage
VariableV2"/device:GPU:1*
shape:?*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
_output_shapes	
:?*
dtype0
?
HCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB10/bias"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
dtype0*
_output_shapes
: 
?
>CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/SwitchSwitchHCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/IsVariableInitializedHCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
_output_shapes
: : 
?
@CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/switch_tIdentity@CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
_output_shapes
: *
T0
*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
?
@CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/switch_fIdentity>CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/Switch"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
_output_shapes
: *
T0

?
?CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/pred_idIdentityHCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0
*
_output_shapes
: 
?
<CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/readIdentityECCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
?
CCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB10/bias?CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0*"
_output_shapes
:?:?
?
@CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/Switch_1Switch+CCN_1Conv_x0/convB10/bias/Initializer/zeros?CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/pred_id*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*"
_output_shapes
:?:?*
T0
?
=CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/MergeMerge@CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/Switch_1<CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/read"/device:GPU:1*
_output_shapes
	:?: *,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
N*
T0
?
Econd_3/read/Switch_CCN_1Conv_x0/convB10/bias/ExponentialMovingAverageSwitch=CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/Mergecond_3/pred_id"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0*"
_output_shapes
:?:?
?
>cond_3/read_CCN_1Conv_x0/convB10/bias/ExponentialMovingAverageIdentityGcond_3/read/Switch_CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage:1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0*
_output_shapes	
:?
?
?cond_3/Merge_CCN_1Conv_x0/convB10/bias/ExponentialMovingAverageMergecond_3/Switch_1>cond_3/read_CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
N*
T0*
_output_shapes
	:?: 
?
9CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/AssignAssign2CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage?cond_3/Merge_CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
_output_shapes	
:?
?
7CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/readIdentity2CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
?
IsVariableInitialized_4IsVariableInitializedCCN_1Conv_x0/convB20/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
_output_shapes
: *
dtype0
{
cond_4/SwitchSwitchIsVariableInitialized_4IsVariableInitialized_4"/device:CPU:0*
_output_shapes
: : *
T0

\
cond_4/switch_tIdentitycond_4/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
Z
cond_4/switch_fIdentitycond_4/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
c
cond_4/pred_idIdentityIsVariableInitialized_4"/device:CPU:0*
_output_shapes
: *
T0

k
cond_4/readIdentitycond_4/read/Switch:1"/device:CPU:0*$
_output_shapes
:??*
T0
?
cond_4/read/Switch	RefSwitchCCN_1Conv_x0/convB20/kernelcond_4/pred_id"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*4
_output_shapes"
 :??:??
?
cond_4/Switch_1Switch6CCN_1Conv_x0/convB20/kernel/Initializer/random_uniformcond_4/pred_id*
T0*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
|
cond_4/MergeMergecond_4/Switch_1cond_4/read"/device:CPU:0*
N*&
_output_shapes
:??: *
T0
?
4CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage
VariableV2"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
shape:??*$
_output_shapes
:??*
dtype0
?
JCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB20/kernel"/device:GPU:1*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
dtype0
?
@CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/SwitchSwitchJCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/IsVariableInitializedJCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_output_shapes
: : *.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
BCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/switch_tIdentityBCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0

?
BCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/switch_fIdentity@CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
_output_shapes
: *
T0
*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
ACCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/pred_idIdentityJCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0

?
>CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/readIdentityGCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0*$
_output_shapes
:??
?
ECCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB20/kernelACCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*4
_output_shapes"
 :??:??*
T0
?
BCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/Switch_1Switch6CCN_1Conv_x0/convB20/kernel/Initializer/random_uniformACCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/pred_id*
T0*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
?CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/MergeMergeBCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/Switch_1>CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/read"/device:GPU:1*&
_output_shapes
:??: *
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
N
?
Gcond_4/read/Switch_CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverageSwitch?CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/Mergecond_4/pred_id"/device:GPU:1*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0
?
@cond_4/read_CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverageIdentityIcond_4/read/Switch_CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage:1"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??
?
Acond_4/Merge_CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverageMergecond_4/Switch_1@cond_4/read_CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage"/device:GPU:1*
T0*&
_output_shapes
:??: *
N*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
;CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/AssignAssign4CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverageAcond_4/Merge_CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
9CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/readIdentity4CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0
?
IsVariableInitialized_5IsVariableInitializedCCN_1Conv_x0/convB20/bias"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
_output_shapes
: *
dtype0
{
cond_5/SwitchSwitchIsVariableInitialized_5IsVariableInitialized_5"/device:CPU:0*
_output_shapes
: : *
T0

\
cond_5/switch_tIdentitycond_5/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
Z
cond_5/switch_fIdentitycond_5/Switch"/device:CPU:0*
_output_shapes
: *
T0

c
cond_5/pred_idIdentityIsVariableInitialized_5"/device:CPU:0*
_output_shapes
: *
T0

b
cond_5/readIdentitycond_5/read/Switch:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
cond_5/read/Switch	RefSwitchCCN_1Conv_x0/convB20/biascond_5/pred_id"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*"
_output_shapes
:?:?
?
cond_5/Switch_1Switch+CCN_1Conv_x0/convB20/bias/Initializer/zeroscond_5/pred_id*
T0*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
s
cond_5/MergeMergecond_5/Switch_1cond_5/read"/device:CPU:0*
N*
_output_shapes
	:?: *
T0
?
2CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage
VariableV2"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
dtype0*
_output_shapes	
:?*
shape:?
?
HCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB20/bias"/device:GPU:1*
dtype0*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
>CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/SwitchSwitchHCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/IsVariableInitializedHCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
_output_shapes
: : 
?
@CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/switch_tIdentity@CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
T0
*
_output_shapes
: 
?
@CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/switch_fIdentity>CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
_output_shapes
: *
T0
*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
?CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/pred_idIdentityHCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
<CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/readIdentityECCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
T0
?
CCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB20/bias?CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*
T0*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
@CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/Switch_1Switch+CCN_1Conv_x0/convB20/bias/Initializer/zeros?CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/pred_id*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
T0
?
=CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/MergeMerge@CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/Switch_1<CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/read"/device:GPU:1*
T0*
_output_shapes
	:?: *
N*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
Econd_5/read/Switch_CCN_1Conv_x0/convB20/bias/ExponentialMovingAverageSwitch=CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/Mergecond_5/pred_id"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*"
_output_shapes
:?:?
?
>cond_5/read_CCN_1Conv_x0/convB20/bias/ExponentialMovingAverageIdentityGcond_5/read/Switch_CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage:1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
_output_shapes	
:?*
T0
?
?cond_5/Merge_CCN_1Conv_x0/convB20/bias/ExponentialMovingAverageMergecond_5/Switch_1>cond_5/read_CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage"/device:GPU:1*
N*
_output_shapes
	:?: *,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
T0
?
9CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/AssignAssign2CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage?cond_5/Merge_CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
_output_shapes	
:?
?
7CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/readIdentity2CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
T0
?
IsVariableInitialized_6IsVariableInitializedCCN_1Conv_x0/convA11/kernel"/device:GPU:1*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
dtype0
{
cond_6/SwitchSwitchIsVariableInitialized_6IsVariableInitialized_6"/device:CPU:0*
T0
*
_output_shapes
: : 
\
cond_6/switch_tIdentitycond_6/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

Z
cond_6/switch_fIdentitycond_6/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
c
cond_6/pred_idIdentityIsVariableInitialized_6"/device:CPU:0*
T0
*
_output_shapes
: 
k
cond_6/readIdentitycond_6/read/Switch:1"/device:CPU:0*
T0*$
_output_shapes
:??
?
cond_6/read/Switch	RefSwitchCCN_1Conv_x0/convA11/kernelcond_6/pred_id"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0*4
_output_shapes"
 :??:??
?
cond_6/Switch_1Switch6CCN_1Conv_x0/convA11/kernel/Initializer/random_uniformcond_6/pred_id*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*4
_output_shapes"
 :??:??
|
cond_6/MergeMergecond_6/Switch_1cond_6/read"/device:CPU:0*
T0*&
_output_shapes
:??: *
N
?
4CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage
VariableV2"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
shape:??*
dtype0
?
JCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convA11/kernel"/device:GPU:1*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
_output_shapes
: 
?
@CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/SwitchSwitchJCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/IsVariableInitializedJCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_output_shapes
: : *.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
BCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/switch_tIdentityBCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0

?
BCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/switch_fIdentity@CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0

?
ACCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/pred_idIdentityJCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0
*
_output_shapes
: 
?
>CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/readIdentityGCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??*
T0
?
ECCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convA11/kernelACCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*4
_output_shapes"
 :??:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
BCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/Switch_1Switch6CCN_1Conv_x0/convA11/kernel/Initializer/random_uniformACCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/pred_id*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0*4
_output_shapes"
 :??:??
?
?CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/MergeMergeBCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/Switch_1>CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/read"/device:GPU:1*
N*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0*&
_output_shapes
:??: 
?
Gcond_6/read/Switch_CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverageSwitch?CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/Mergecond_6/pred_id"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*4
_output_shapes"
 :??:??
?
@cond_6/read_CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverageIdentityIcond_6/read/Switch_CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage:1"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0*$
_output_shapes
:??
?
Acond_6/Merge_CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverageMergecond_6/Switch_1@cond_6/read_CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage"/device:GPU:1*&
_output_shapes
:??: *
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
N
?
;CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/AssignAssign4CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverageAcond_6/Merge_CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0*$
_output_shapes
:??
?
9CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/readIdentity4CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??*
T0
?
IsVariableInitialized_7IsVariableInitializedCCN_1Conv_x0/convA11/bias"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes
: *
dtype0
{
cond_7/SwitchSwitchIsVariableInitialized_7IsVariableInitialized_7"/device:CPU:0*
_output_shapes
: : *
T0

\
cond_7/switch_tIdentitycond_7/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
Z
cond_7/switch_fIdentitycond_7/Switch"/device:CPU:0*
_output_shapes
: *
T0

c
cond_7/pred_idIdentityIsVariableInitialized_7"/device:CPU:0*
T0
*
_output_shapes
: 
b
cond_7/readIdentitycond_7/read/Switch:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
cond_7/read/Switch	RefSwitchCCN_1Conv_x0/convA11/biascond_7/pred_id"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*"
_output_shapes
:?:?*
T0
?
cond_7/Switch_1Switch+CCN_1Conv_x0/convA11/bias/Initializer/zeroscond_7/pred_id*"
_output_shapes
:?:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias
s
cond_7/MergeMergecond_7/Switch_1cond_7/read"/device:CPU:0*
_output_shapes
	:?: *
T0*
N
?
2CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage
VariableV2"/device:GPU:1*
dtype0*
shape:?*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?
?
HCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convA11/bias"/device:GPU:1*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
dtype0
?
>CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/SwitchSwitchHCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/IsVariableInitializedHCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes
: : 
?
@CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/switch_tIdentity@CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0

?
@CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/switch_fIdentity>CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
T0
*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convA11/bias
?
?CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/pred_idIdentityHCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convA11/bias
?
<CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/readIdentityECCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0
?
CCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convA11/bias?CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*"
_output_shapes
:?:?
?
@CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/Switch_1Switch+CCN_1Conv_x0/convA11/bias/Initializer/zeros?CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/pred_id*"
_output_shapes
:?:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias
?
=CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/MergeMerge@CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/Switch_1<CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/read"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
N*
T0*
_output_shapes
	:?: 
?
Econd_7/read/Switch_CCN_1Conv_x0/convA11/bias/ExponentialMovingAverageSwitch=CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/Mergecond_7/pred_id"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0*"
_output_shapes
:?:?
?
>cond_7/read_CCN_1Conv_x0/convA11/bias/ExponentialMovingAverageIdentityGcond_7/read/Switch_CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage:1"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?
?
?cond_7/Merge_CCN_1Conv_x0/convA11/bias/ExponentialMovingAverageMergecond_7/Switch_1>cond_7/read_CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage"/device:GPU:1*
N*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes
	:?: *
T0
?
9CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/AssignAssign2CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage?cond_7/Merge_CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0
?
7CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/readIdentity2CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0*
_output_shapes	
:?
?
IsVariableInitialized_8IsVariableInitializedCCN_1Conv_x0/convB11/kernel"/device:GPU:1*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
_output_shapes
: 
{
cond_8/SwitchSwitchIsVariableInitialized_8IsVariableInitialized_8"/device:CPU:0*
_output_shapes
: : *
T0

\
cond_8/switch_tIdentitycond_8/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
Z
cond_8/switch_fIdentitycond_8/Switch"/device:CPU:0*
_output_shapes
: *
T0

c
cond_8/pred_idIdentityIsVariableInitialized_8"/device:CPU:0*
_output_shapes
: *
T0

k
cond_8/readIdentitycond_8/read/Switch:1"/device:CPU:0*
T0*$
_output_shapes
:??
?
cond_8/read/Switch	RefSwitchCCN_1Conv_x0/convB11/kernelcond_8/pred_id"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*4
_output_shapes"
 :??:??*
T0
?
cond_8/Switch_1Switch6CCN_1Conv_x0/convB11/kernel/Initializer/random_uniformcond_8/pred_id*
T0*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
|
cond_8/MergeMergecond_8/Switch_1cond_8/read"/device:CPU:0*
N*&
_output_shapes
:??: *
T0
?
4CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage
VariableV2"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
shape:??*
dtype0
?
JCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB11/kernel"/device:GPU:1*
dtype0*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
@CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/SwitchSwitchJCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/IsVariableInitializedJCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_output_shapes
: : *.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
BCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/switch_tIdentityBCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
_output_shapes
: *
T0

?
BCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/switch_fIdentity@CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0

?
ACCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/pred_idIdentityJCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: *
T0
*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
>CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/readIdentityGCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0
?
ECCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB11/kernelACCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*4
_output_shapes"
 :??:??*
T0
?
BCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/Switch_1Switch6CCN_1Conv_x0/convB11/kernel/Initializer/random_uniformACCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/pred_id*
T0*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
?CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/MergeMergeBCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/Switch_1>CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/read"/device:GPU:1*&
_output_shapes
:??: *
T0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
N
?
Gcond_8/read/Switch_CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverageSwitch?CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/Mergecond_8/pred_id"/device:GPU:1*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0
?
@cond_8/read_CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverageIdentityIcond_8/read/Switch_CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage:1"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*$
_output_shapes
:??*
T0
?
Acond_8/Merge_CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverageMergecond_8/Switch_1@cond_8/read_CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage"/device:GPU:1*&
_output_shapes
:??: *
T0*
N*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
;CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/AssignAssign4CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverageAcond_8/Merge_CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0
?
9CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/readIdentity4CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*$
_output_shapes
:??
?
IsVariableInitialized_9IsVariableInitializedCCN_1Conv_x0/convB11/bias"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes
: *
dtype0
{
cond_9/SwitchSwitchIsVariableInitialized_9IsVariableInitialized_9"/device:CPU:0*
T0
*
_output_shapes
: : 
\
cond_9/switch_tIdentitycond_9/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

Z
cond_9/switch_fIdentitycond_9/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
c
cond_9/pred_idIdentityIsVariableInitialized_9"/device:CPU:0*
T0
*
_output_shapes
: 
b
cond_9/readIdentitycond_9/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
cond_9/read/Switch	RefSwitchCCN_1Conv_x0/convB11/biascond_9/pred_id"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
T0*"
_output_shapes
:?:?
?
cond_9/Switch_1Switch+CCN_1Conv_x0/convB11/bias/Initializer/zeroscond_9/pred_id*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*"
_output_shapes
:?:?
s
cond_9/MergeMergecond_9/Switch_1cond_9/read"/device:CPU:0*
N*
_output_shapes
	:?: *
T0
?
2CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage
VariableV2"/device:GPU:1*
dtype0*
shape:?*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?
?
HCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB11/bias"/device:GPU:1*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
dtype0
?
>CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/SwitchSwitchHCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/IsVariableInitializedHCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: : *
T0
*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
@CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/switch_tIdentity@CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
T0
*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
@CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/switch_fIdentity>CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
T0
*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes
: 
?
?CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/pred_idIdentityHCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
T0
*
_output_shapes
: 
?
<CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/readIdentityECCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
CCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB11/bias?CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*"
_output_shapes
:?:?
?
@CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/Switch_1Switch+CCN_1Conv_x0/convB11/bias/Initializer/zeros?CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/pred_id*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*"
_output_shapes
:?:?*
T0
?
=CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/MergeMerge@CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/Switch_1<CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/read"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes
	:?: *
T0*
N
?
Econd_9/read/Switch_CCN_1Conv_x0/convB11/bias/ExponentialMovingAverageSwitch=CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/Mergecond_9/pred_id"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*"
_output_shapes
:?:?
?
>cond_9/read_CCN_1Conv_x0/convB11/bias/ExponentialMovingAverageIdentityGcond_9/read/Switch_CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage:1"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
?cond_9/Merge_CCN_1Conv_x0/convB11/bias/ExponentialMovingAverageMergecond_9/Switch_1>cond_9/read_CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage"/device:GPU:1*
N*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes
	:?: 
?
9CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/AssignAssign2CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage?cond_9/Merge_CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?
?
7CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/readIdentity2CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
IsVariableInitialized_10IsVariableInitializedCCN_1Conv_x0/convB21/kernel"/device:GPU:1*
_output_shapes
: *
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
~
cond_10/SwitchSwitchIsVariableInitialized_10IsVariableInitialized_10"/device:CPU:0*
_output_shapes
: : *
T0

^
cond_10/switch_tIdentitycond_10/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

\
cond_10/switch_fIdentitycond_10/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
e
cond_10/pred_idIdentityIsVariableInitialized_10"/device:CPU:0*
_output_shapes
: *
T0

m
cond_10/readIdentitycond_10/read/Switch:1"/device:CPU:0*$
_output_shapes
:??*
T0
?
cond_10/read/Switch	RefSwitchCCN_1Conv_x0/convB21/kernelcond_10/pred_id"/device:GPU:1*4
_output_shapes"
 :??:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
cond_10/Switch_1Switch6CCN_1Conv_x0/convB21/kernel/Initializer/random_uniformcond_10/pred_id*4
_output_shapes"
 :??:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel

cond_10/MergeMergecond_10/Switch_1cond_10/read"/device:CPU:0*&
_output_shapes
:??: *
T0*
N
?
4CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage
VariableV2"/device:GPU:1*
shape:??*
dtype0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
JCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB21/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
_output_shapes
: *
dtype0
?
@CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/SwitchSwitchJCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/IsVariableInitializedJCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
T0
*
_output_shapes
: : 
?
BCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/switch_tIdentityBCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
_output_shapes
: *
T0
*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
BCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/switch_fIdentity@CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
T0
*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
_output_shapes
: 
?
ACCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/pred_idIdentityJCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
>CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/readIdentityGCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??*
T0
?
ECCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB21/kernelACCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*
T0*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
BCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/Switch_1Switch6CCN_1Conv_x0/convB21/kernel/Initializer/random_uniformACCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/pred_id*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*4
_output_shapes"
 :??:??*
T0
?
?CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/MergeMergeBCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/Switch_1>CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/read"/device:GPU:1*
T0*&
_output_shapes
:??: *.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
N
?
Hcond_10/read/Switch_CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverageSwitch?CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/Mergecond_10/pred_id"/device:GPU:1*4
_output_shapes"
 :??:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
Acond_10/read_CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverageIdentityJcond_10/read/Switch_CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage:1"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??*
T0
?
Bcond_10/Merge_CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverageMergecond_10/Switch_1Acond_10/read_CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage"/device:GPU:1*
N*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*&
_output_shapes
:??: *
T0
?
;CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/AssignAssign4CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverageBcond_10/Merge_CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??*
T0
?
9CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/readIdentity4CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??
?
IsVariableInitialized_11IsVariableInitializedCCN_1Conv_x0/convB21/bias"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes
: *
dtype0
~
cond_11/SwitchSwitchIsVariableInitialized_11IsVariableInitialized_11"/device:CPU:0*
_output_shapes
: : *
T0

^
cond_11/switch_tIdentitycond_11/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

\
cond_11/switch_fIdentitycond_11/Switch"/device:CPU:0*
_output_shapes
: *
T0

e
cond_11/pred_idIdentityIsVariableInitialized_11"/device:CPU:0*
T0
*
_output_shapes
: 
d
cond_11/readIdentitycond_11/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
cond_11/read/Switch	RefSwitchCCN_1Conv_x0/convB21/biascond_11/pred_id"/device:GPU:1*"
_output_shapes
:?:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
cond_11/Switch_1Switch+CCN_1Conv_x0/convB21/bias/Initializer/zeroscond_11/pred_id*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0
v
cond_11/MergeMergecond_11/Switch_1cond_11/read"/device:CPU:0*
N*
_output_shapes
	:?: *
T0
?
2CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage
VariableV2"/device:GPU:1*
dtype0*
shape:?*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
HCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB21/bias"/device:GPU:1*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes
: 
?
>CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/SwitchSwitchHCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/IsVariableInitializedHCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: : *
T0
*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
@CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/switch_tIdentity@CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
T0
*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes
: 
?
@CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/switch_fIdentity>CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
T0
*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
?CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/pred_idIdentityHCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes
: 
?
<CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/readIdentityECCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes	
:?*
T0
?
CCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB21/bias?CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*"
_output_shapes
:?:?*
T0
?
@CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/Switch_1Switch+CCN_1Conv_x0/convB21/bias/Initializer/zeros?CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/pred_id*
T0*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
=CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/MergeMerge@CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/Switch_1<CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/read"/device:GPU:1*
T0*
N*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes
	:?: 
?
Fcond_11/read/Switch_CCN_1Conv_x0/convB21/bias/ExponentialMovingAverageSwitch=CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/Mergecond_11/pred_id"/device:GPU:1*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0
?
?cond_11/read_CCN_1Conv_x0/convB21/bias/ExponentialMovingAverageIdentityHcond_11/read/Switch_CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage:1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0*
_output_shapes	
:?
?
@cond_11/Merge_CCN_1Conv_x0/convB21/bias/ExponentialMovingAverageMergecond_11/Switch_1?cond_11/read_CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
N*
_output_shapes
	:?: *
T0
?
9CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/AssignAssign2CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage@cond_11/Merge_CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
7CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/readIdentity2CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0*
_output_shapes	
:?
?
IsVariableInitialized_12IsVariableInitializedConv_out__/beta"/device:GPU:1*
dtype0*"
_class
loc:@Conv_out__/beta*
_output_shapes
: 
~
cond_12/SwitchSwitchIsVariableInitialized_12IsVariableInitialized_12"/device:CPU:0*
T0
*
_output_shapes
: : 
^
cond_12/switch_tIdentitycond_12/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

\
cond_12/switch_fIdentitycond_12/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
e
cond_12/pred_idIdentityIsVariableInitialized_12"/device:CPU:0*
_output_shapes
: *
T0

d
cond_12/readIdentitycond_12/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
cond_12/read/Switch	RefSwitchConv_out__/betacond_12/pred_id"/device:GPU:1*"
_class
loc:@Conv_out__/beta*"
_output_shapes
:?:?*
T0
?
cond_12/Switch_1Switch!Conv_out__/beta/Initializer/zeroscond_12/pred_id*
T0*"
_class
loc:@Conv_out__/beta*"
_output_shapes
:?:?
v
cond_12/MergeMergecond_12/Switch_1cond_12/read"/device:CPU:0*
_output_shapes
	:?: *
N*
T0
?
(Conv_out__/beta/ExponentialMovingAverage
VariableV2"/device:GPU:1*
_output_shapes	
:?*
shape:?*
dtype0*"
_class
loc:@Conv_out__/beta
?
>Conv_out__/beta/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedConv_out__/beta"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
dtype0*
_output_shapes
: 
?
4Conv_out__/beta/ExponentialMovingAverage/cond/SwitchSwitch>Conv_out__/beta/ExponentialMovingAverage/IsVariableInitialized>Conv_out__/beta/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
_output_shapes
: : *
T0

?
6Conv_out__/beta/ExponentialMovingAverage/cond/switch_tIdentity6Conv_out__/beta/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
T0
*
_output_shapes
: *"
_class
loc:@Conv_out__/beta
?
6Conv_out__/beta/ExponentialMovingAverage/cond/switch_fIdentity4Conv_out__/beta/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
_output_shapes
: *"
_class
loc:@Conv_out__/beta*
T0

?
5Conv_out__/beta/ExponentialMovingAverage/cond/pred_idIdentity>Conv_out__/beta/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: *"
_class
loc:@Conv_out__/beta*
T0

?
2Conv_out__/beta/ExponentialMovingAverage/cond/readIdentity;Conv_out__/beta/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
_output_shapes	
:?*
T0*"
_class
loc:@Conv_out__/beta
?
9Conv_out__/beta/ExponentialMovingAverage/cond/read/Switch	RefSwitchConv_out__/beta5Conv_out__/beta/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
T0*"
_output_shapes
:?:?
?
6Conv_out__/beta/ExponentialMovingAverage/cond/Switch_1Switch!Conv_out__/beta/Initializer/zeros5Conv_out__/beta/ExponentialMovingAverage/cond/pred_id*"
_class
loc:@Conv_out__/beta*
T0*"
_output_shapes
:?:?
?
3Conv_out__/beta/ExponentialMovingAverage/cond/MergeMerge6Conv_out__/beta/ExponentialMovingAverage/cond/Switch_12Conv_out__/beta/ExponentialMovingAverage/cond/read"/device:GPU:1*
T0*
N*
_output_shapes
	:?: *"
_class
loc:@Conv_out__/beta
?
<cond_12/read/Switch_Conv_out__/beta/ExponentialMovingAverageSwitch3Conv_out__/beta/ExponentialMovingAverage/cond/Mergecond_12/pred_id"/device:GPU:1*"
_output_shapes
:?:?*"
_class
loc:@Conv_out__/beta*
T0
?
5cond_12/read_Conv_out__/beta/ExponentialMovingAverageIdentity>cond_12/read/Switch_Conv_out__/beta/ExponentialMovingAverage:1"/device:GPU:1*
T0*
_output_shapes	
:?*"
_class
loc:@Conv_out__/beta
?
6cond_12/Merge_Conv_out__/beta/ExponentialMovingAverageMergecond_12/Switch_15cond_12/read_Conv_out__/beta/ExponentialMovingAverage"/device:GPU:1*
T0*"
_class
loc:@Conv_out__/beta*
_output_shapes
	:?: *
N
?
/Conv_out__/beta/ExponentialMovingAverage/AssignAssign(Conv_out__/beta/ExponentialMovingAverage6cond_12/Merge_Conv_out__/beta/ExponentialMovingAverage"/device:GPU:1*
T0*
_output_shapes	
:?*"
_class
loc:@Conv_out__/beta
?
-Conv_out__/beta/ExponentialMovingAverage/readIdentity(Conv_out__/beta/ExponentialMovingAverage"/device:GPU:1*
T0*"
_class
loc:@Conv_out__/beta*
_output_shapes	
:?
?
IsVariableInitialized_13IsVariableInitializedConv_out__/gamma"/device:GPU:1*
dtype0*
_output_shapes
: *#
_class
loc:@Conv_out__/gamma
~
cond_13/SwitchSwitchIsVariableInitialized_13IsVariableInitialized_13"/device:CPU:0*
T0
*
_output_shapes
: : 
^
cond_13/switch_tIdentitycond_13/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
\
cond_13/switch_fIdentitycond_13/Switch"/device:CPU:0*
_output_shapes
: *
T0

e
cond_13/pred_idIdentityIsVariableInitialized_13"/device:CPU:0*
_output_shapes
: *
T0

d
cond_13/readIdentitycond_13/read/Switch:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
cond_13/read/Switch	RefSwitchConv_out__/gammacond_13/pred_id"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
T0*"
_output_shapes
:?:?
?
cond_13/Switch_1Switch!Conv_out__/gamma/Initializer/onescond_13/pred_id*"
_output_shapes
:?:?*
T0*#
_class
loc:@Conv_out__/gamma
v
cond_13/MergeMergecond_13/Switch_1cond_13/read"/device:CPU:0*
_output_shapes
	:?: *
N*
T0
?
)Conv_out__/gamma/ExponentialMovingAverage
VariableV2"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
dtype0*
_output_shapes	
:?*
shape:?
?
?Conv_out__/gamma/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedConv_out__/gamma"/device:GPU:1*
dtype0*#
_class
loc:@Conv_out__/gamma*
_output_shapes
: 
?
5Conv_out__/gamma/ExponentialMovingAverage/cond/SwitchSwitch?Conv_out__/gamma/ExponentialMovingAverage/IsVariableInitialized?Conv_out__/gamma/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: : *#
_class
loc:@Conv_out__/gamma*
T0

?
7Conv_out__/gamma/ExponentialMovingAverage/cond/switch_tIdentity7Conv_out__/gamma/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
_output_shapes
: *
T0

?
7Conv_out__/gamma/ExponentialMovingAverage/cond/switch_fIdentity5Conv_out__/gamma/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
T0
*
_output_shapes
: *#
_class
loc:@Conv_out__/gamma
?
6Conv_out__/gamma/ExponentialMovingAverage/cond/pred_idIdentity?Conv_out__/gamma/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_output_shapes
: *#
_class
loc:@Conv_out__/gamma
?
3Conv_out__/gamma/ExponentialMovingAverage/cond/readIdentity<Conv_out__/gamma/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
_output_shapes	
:?*
T0
?
:Conv_out__/gamma/ExponentialMovingAverage/cond/read/Switch	RefSwitchConv_out__/gamma6Conv_out__/gamma/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*"
_output_shapes
:?:?*
T0*#
_class
loc:@Conv_out__/gamma
?
7Conv_out__/gamma/ExponentialMovingAverage/cond/Switch_1Switch!Conv_out__/gamma/Initializer/ones6Conv_out__/gamma/ExponentialMovingAverage/cond/pred_id*"
_output_shapes
:?:?*
T0*#
_class
loc:@Conv_out__/gamma
?
4Conv_out__/gamma/ExponentialMovingAverage/cond/MergeMerge7Conv_out__/gamma/ExponentialMovingAverage/cond/Switch_13Conv_out__/gamma/ExponentialMovingAverage/cond/read"/device:GPU:1*
_output_shapes
	:?: *
T0*
N*#
_class
loc:@Conv_out__/gamma
?
=cond_13/read/Switch_Conv_out__/gamma/ExponentialMovingAverageSwitch4Conv_out__/gamma/ExponentialMovingAverage/cond/Mergecond_13/pred_id"/device:GPU:1*"
_output_shapes
:?:?*
T0*#
_class
loc:@Conv_out__/gamma
?
6cond_13/read_Conv_out__/gamma/ExponentialMovingAverageIdentity?cond_13/read/Switch_Conv_out__/gamma/ExponentialMovingAverage:1"/device:GPU:1*
T0*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma
?
7cond_13/Merge_Conv_out__/gamma/ExponentialMovingAverageMergecond_13/Switch_16cond_13/read_Conv_out__/gamma/ExponentialMovingAverage"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
T0*
_output_shapes
	:?: *
N
?
0Conv_out__/gamma/ExponentialMovingAverage/AssignAssign)Conv_out__/gamma/ExponentialMovingAverage7cond_13/Merge_Conv_out__/gamma/ExponentialMovingAverage"/device:GPU:1*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma*
T0
?
.Conv_out__/gamma/ExponentialMovingAverage/readIdentity)Conv_out__/gamma/ExponentialMovingAverage"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
_output_shapes	
:?*
T0
?
IsVariableInitialized_14IsVariableInitialized"Reconstruction_Output/dense/kernel"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
dtype0*
_output_shapes
: 
~
cond_14/SwitchSwitchIsVariableInitialized_14IsVariableInitialized_14"/device:CPU:0*
_output_shapes
: : *
T0

^
cond_14/switch_tIdentitycond_14/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
\
cond_14/switch_fIdentitycond_14/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
e
cond_14/pred_idIdentityIsVariableInitialized_14"/device:CPU:0*
T0
*
_output_shapes
: 
h
cond_14/readIdentitycond_14/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
:	?
?
cond_14/read/Switch	RefSwitch"Reconstruction_Output/dense/kernelcond_14/pred_id"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel**
_output_shapes
:	?:	?*
T0
?
cond_14/Switch_1Switch=Reconstruction_Output/dense/kernel/Initializer/random_uniformcond_14/pred_id*
T0**
_output_shapes
:	?:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
z
cond_14/MergeMergecond_14/Switch_1cond_14/read"/device:CPU:0*
T0*
N*!
_output_shapes
:	?: 
?
;Reconstruction_Output/dense/kernel/ExponentialMovingAverage
VariableV2"/device:GPU:1*
_output_shapes
:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
shape:	?*
dtype0
?
QReconstruction_Output/dense/kernel/ExponentialMovingAverage/IsVariableInitializedIsVariableInitialized"Reconstruction_Output/dense/kernel"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
: *
dtype0
?
GReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/SwitchSwitchQReconstruction_Output/dense/kernel/ExponentialMovingAverage/IsVariableInitializedQReconstruction_Output/dense/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: : *
T0
*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
IReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/switch_tIdentityIReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
T0
*
_output_shapes
: 
?
IReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/switch_fIdentityGReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
_output_shapes
: *
T0
*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
HReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/pred_idIdentityQReconstruction_Output/dense/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
: *
T0

?
EReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/readIdentityNReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
_output_shapes
:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
T0
?
LReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/read/Switch	RefSwitch"Reconstruction_Output/dense/kernelHReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel**
_output_shapes
:	?:	?*
T0
?
IReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/Switch_1Switch=Reconstruction_Output/dense/kernel/Initializer/random_uniformHReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/pred_id**
_output_shapes
:	?:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
T0
?
FReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/MergeMergeIReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/Switch_1EReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/read"/device:GPU:1*
T0*!
_output_shapes
:	?: *5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
N
?
Ocond_14/read/Switch_Reconstruction_Output/dense/kernel/ExponentialMovingAverageSwitchFReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/Mergecond_14/pred_id"/device:GPU:1**
_output_shapes
:	?:	?*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
Hcond_14/read_Reconstruction_Output/dense/kernel/ExponentialMovingAverageIdentityQcond_14/read/Switch_Reconstruction_Output/dense/kernel/ExponentialMovingAverage:1"/device:GPU:1*
T0*
_output_shapes
:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
Icond_14/Merge_Reconstruction_Output/dense/kernel/ExponentialMovingAverageMergecond_14/Switch_1Hcond_14/read_Reconstruction_Output/dense/kernel/ExponentialMovingAverage"/device:GPU:1*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*!
_output_shapes
:	?: *
N
?
BReconstruction_Output/dense/kernel/ExponentialMovingAverage/AssignAssign;Reconstruction_Output/dense/kernel/ExponentialMovingAverageIcond_14/Merge_Reconstruction_Output/dense/kernel/ExponentialMovingAverage"/device:GPU:1*
_output_shapes
:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
T0
?
@Reconstruction_Output/dense/kernel/ExponentialMovingAverage/readIdentity;Reconstruction_Output/dense/kernel/ExponentialMovingAverage"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
T0*
_output_shapes
:	?
?
IsVariableInitialized_15IsVariableInitialized Reconstruction_Output/dense/bias"/device:GPU:1*
_output_shapes
: *
dtype0*3
_class)
'%loc:@Reconstruction_Output/dense/bias
~
cond_15/SwitchSwitchIsVariableInitialized_15IsVariableInitialized_15"/device:CPU:0*
_output_shapes
: : *
T0

^
cond_15/switch_tIdentitycond_15/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
\
cond_15/switch_fIdentitycond_15/Switch"/device:CPU:0*
_output_shapes
: *
T0

e
cond_15/pred_idIdentityIsVariableInitialized_15"/device:CPU:0*
_output_shapes
: *
T0

c
cond_15/readIdentitycond_15/read/Switch:1"/device:CPU:0*
_output_shapes
:*
T0
?
cond_15/read/Switch	RefSwitch Reconstruction_Output/dense/biascond_15/pred_id"/device:GPU:1*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias* 
_output_shapes
::
?
cond_15/Switch_1Switch2Reconstruction_Output/dense/bias/Initializer/zeroscond_15/pred_id*3
_class)
'%loc:@Reconstruction_Output/dense/bias* 
_output_shapes
::*
T0
u
cond_15/MergeMergecond_15/Switch_1cond_15/read"/device:CPU:0*
_output_shapes

:: *
N*
T0
?
9Reconstruction_Output/dense/bias/ExponentialMovingAverage
VariableV2"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
:*
shape:*
dtype0
?
OReconstruction_Output/dense/bias/ExponentialMovingAverage/IsVariableInitializedIsVariableInitialized Reconstruction_Output/dense/bias"/device:GPU:1*
dtype0*
_output_shapes
: *3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
EReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/SwitchSwitchOReconstruction_Output/dense/bias/ExponentialMovingAverage/IsVariableInitializedOReconstruction_Output/dense/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_output_shapes
: : *3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
GReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/switch_tIdentityGReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
: *
T0

?
GReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/switch_fIdentityEReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
_output_shapes
: *
T0
*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
FReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/pred_idIdentityOReconstruction_Output/dense/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
T0
*
_output_shapes
: 
?
CReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/readIdentityLReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
_output_shapes
:*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
JReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/read/Switch	RefSwitch Reconstruction_Output/dense/biasFReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias* 
_output_shapes
::
?
GReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/Switch_1Switch2Reconstruction_Output/dense/bias/Initializer/zerosFReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/pred_id*3
_class)
'%loc:@Reconstruction_Output/dense/bias* 
_output_shapes
::*
T0
?
DReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/MergeMergeGReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/Switch_1CReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/read"/device:GPU:1*
N*
_output_shapes

:: *3
_class)
'%loc:@Reconstruction_Output/dense/bias*
T0
?
Mcond_15/read/Switch_Reconstruction_Output/dense/bias/ExponentialMovingAverageSwitchDReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/Mergecond_15/pred_id"/device:GPU:1*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias* 
_output_shapes
::
?
Fcond_15/read_Reconstruction_Output/dense/bias/ExponentialMovingAverageIdentityOcond_15/read/Switch_Reconstruction_Output/dense/bias/ExponentialMovingAverage:1"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
:*
T0
?
Gcond_15/Merge_Reconstruction_Output/dense/bias/ExponentialMovingAverageMergecond_15/Switch_1Fcond_15/read_Reconstruction_Output/dense/bias/ExponentialMovingAverage"/device:GPU:1*
T0*
_output_shapes

:: *3
_class)
'%loc:@Reconstruction_Output/dense/bias*
N
?
@Reconstruction_Output/dense/bias/ExponentialMovingAverage/AssignAssign9Reconstruction_Output/dense/bias/ExponentialMovingAverageGcond_15/Merge_Reconstruction_Output/dense/bias/ExponentialMovingAverage"/device:GPU:1*
_output_shapes
:*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
>Reconstruction_Output/dense/bias/ExponentialMovingAverage/readIdentity9Reconstruction_Output/dense/bias/ExponentialMovingAverage"/device:GPU:1*
_output_shapes
:*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
IsVariableInitialized_16IsVariableInitializeddense/kernel"/device:GPU:1*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0
~
cond_16/SwitchSwitchIsVariableInitialized_16IsVariableInitialized_16"/device:CPU:0*
_output_shapes
: : *
T0

^
cond_16/switch_tIdentitycond_16/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
\
cond_16/switch_fIdentitycond_16/Switch"/device:CPU:0*
_output_shapes
: *
T0

e
cond_16/pred_idIdentityIsVariableInitialized_16"/device:CPU:0*
_output_shapes
: *
T0

h
cond_16/readIdentitycond_16/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
:	? 
?
cond_16/read/Switch	RefSwitchdense/kernelcond_16/pred_id"/device:GPU:1*
_class
loc:@dense/kernel*
T0**
_output_shapes
:	? :	? 
?
cond_16/Switch_1Switch'dense/kernel/Initializer/random_uniformcond_16/pred_id*
T0**
_output_shapes
:	? :	? *
_class
loc:@dense/kernel
z
cond_16/MergeMergecond_16/Switch_1cond_16/read"/device:CPU:0*
T0*!
_output_shapes
:	? : *
N
?
%dense/kernel/ExponentialMovingAverage
VariableV2"/device:GPU:1*
dtype0*
shape:	? *
_class
loc:@dense/kernel*
_output_shapes
:	? 
?
;dense/kernel/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializeddense/kernel"/device:GPU:1*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: 
?
1dense/kernel/ExponentialMovingAverage/cond/SwitchSwitch;dense/kernel/ExponentialMovingAverage/IsVariableInitialized;dense/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_output_shapes
: : *
_class
loc:@dense/kernel
?
3dense/kernel/ExponentialMovingAverage/cond/switch_tIdentity3dense/kernel/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
T0
*
_output_shapes
: *
_class
loc:@dense/kernel
?
3dense/kernel/ExponentialMovingAverage/cond/switch_fIdentity1dense/kernel/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
T0
*
_output_shapes
: *
_class
loc:@dense/kernel
?
2dense/kernel/ExponentialMovingAverage/cond/pred_idIdentity;dense/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_class
loc:@dense/kernel*
T0
*
_output_shapes
: 
?
/dense/kernel/ExponentialMovingAverage/cond/readIdentity8dense/kernel/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	? 
?
6dense/kernel/ExponentialMovingAverage/cond/read/Switch	RefSwitchdense/kernel2dense/kernel/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*
T0**
_output_shapes
:	? :	? *
_class
loc:@dense/kernel
?
3dense/kernel/ExponentialMovingAverage/cond/Switch_1Switch'dense/kernel/Initializer/random_uniform2dense/kernel/ExponentialMovingAverage/cond/pred_id*
T0**
_output_shapes
:	? :	? *
_class
loc:@dense/kernel
?
0dense/kernel/ExponentialMovingAverage/cond/MergeMerge3dense/kernel/ExponentialMovingAverage/cond/Switch_1/dense/kernel/ExponentialMovingAverage/cond/read"/device:GPU:1*!
_output_shapes
:	? : *
_class
loc:@dense/kernel*
N*
T0
?
9cond_16/read/Switch_dense/kernel/ExponentialMovingAverageSwitch0dense/kernel/ExponentialMovingAverage/cond/Mergecond_16/pred_id"/device:GPU:1**
_output_shapes
:	? :	? *
_class
loc:@dense/kernel*
T0
?
2cond_16/read_dense/kernel/ExponentialMovingAverageIdentity;cond_16/read/Switch_dense/kernel/ExponentialMovingAverage:1"/device:GPU:1*
_output_shapes
:	? *
T0*
_class
loc:@dense/kernel
?
3cond_16/Merge_dense/kernel/ExponentialMovingAverageMergecond_16/Switch_12cond_16/read_dense/kernel/ExponentialMovingAverage"/device:GPU:1*
_class
loc:@dense/kernel*!
_output_shapes
:	? : *
N*
T0
?
,dense/kernel/ExponentialMovingAverage/AssignAssign%dense/kernel/ExponentialMovingAverage3cond_16/Merge_dense/kernel/ExponentialMovingAverage"/device:GPU:1*
T0*
_output_shapes
:	? *
_class
loc:@dense/kernel
?
*dense/kernel/ExponentialMovingAverage/readIdentity%dense/kernel/ExponentialMovingAverage"/device:GPU:1*
_class
loc:@dense/kernel*
_output_shapes
:	? *
T0
?
IsVariableInitialized_17IsVariableInitialized
dense/bias"/device:GPU:1*
dtype0*
_output_shapes
: *
_class
loc:@dense/bias
~
cond_17/SwitchSwitchIsVariableInitialized_17IsVariableInitialized_17"/device:CPU:0*
T0
*
_output_shapes
: : 
^
cond_17/switch_tIdentitycond_17/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

\
cond_17/switch_fIdentitycond_17/Switch"/device:CPU:0*
_output_shapes
: *
T0

e
cond_17/pred_idIdentityIsVariableInitialized_17"/device:CPU:0*
T0
*
_output_shapes
: 
c
cond_17/readIdentitycond_17/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
: 
?
cond_17/read/Switch	RefSwitch
dense/biascond_17/pred_id"/device:GPU:1*
_class
loc:@dense/bias* 
_output_shapes
: : *
T0
?
cond_17/Switch_1Switchdense/bias/Initializer/zeroscond_17/pred_id*
T0* 
_output_shapes
: : *
_class
loc:@dense/bias
u
cond_17/MergeMergecond_17/Switch_1cond_17/read"/device:CPU:0*
N*
T0*
_output_shapes

: : 
?
#dense/bias/ExponentialMovingAverage
VariableV2"/device:GPU:1*
_output_shapes
: *
dtype0*
_class
loc:@dense/bias*
shape: 
?
9dense/bias/ExponentialMovingAverage/IsVariableInitializedIsVariableInitialized
dense/bias"/device:GPU:1*
dtype0*
_class
loc:@dense/bias*
_output_shapes
: 
?
/dense/bias/ExponentialMovingAverage/cond/SwitchSwitch9dense/bias/ExponentialMovingAverage/IsVariableInitialized9dense/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_class
loc:@dense/bias*
_output_shapes
: : 
?
1dense/bias/ExponentialMovingAverage/cond/switch_tIdentity1dense/bias/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
_output_shapes
: *
T0
*
_class
loc:@dense/bias
?
1dense/bias/ExponentialMovingAverage/cond/switch_fIdentity/dense/bias/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
T0
*
_class
loc:@dense/bias*
_output_shapes
: 
?
0dense/bias/ExponentialMovingAverage/cond/pred_idIdentity9dense/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_class
loc:@dense/bias*
_output_shapes
: 
?
-dense/bias/ExponentialMovingAverage/cond/readIdentity6dense/bias/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
_class
loc:@dense/bias*
_output_shapes
: *
T0
?
4dense/bias/ExponentialMovingAverage/cond/read/Switch	RefSwitch
dense/bias0dense/bias/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*
_class
loc:@dense/bias*
T0* 
_output_shapes
: : 
?
1dense/bias/ExponentialMovingAverage/cond/Switch_1Switchdense/bias/Initializer/zeros0dense/bias/ExponentialMovingAverage/cond/pred_id*
_class
loc:@dense/bias*
T0* 
_output_shapes
: : 
?
.dense/bias/ExponentialMovingAverage/cond/MergeMerge1dense/bias/ExponentialMovingAverage/cond/Switch_1-dense/bias/ExponentialMovingAverage/cond/read"/device:GPU:1*
T0*
_output_shapes

: : *
N*
_class
loc:@dense/bias
?
7cond_17/read/Switch_dense/bias/ExponentialMovingAverageSwitch.dense/bias/ExponentialMovingAverage/cond/Mergecond_17/pred_id"/device:GPU:1* 
_output_shapes
: : *
T0*
_class
loc:@dense/bias
?
0cond_17/read_dense/bias/ExponentialMovingAverageIdentity9cond_17/read/Switch_dense/bias/ExponentialMovingAverage:1"/device:GPU:1*
_output_shapes
: *
_class
loc:@dense/bias*
T0
?
1cond_17/Merge_dense/bias/ExponentialMovingAverageMergecond_17/Switch_10cond_17/read_dense/bias/ExponentialMovingAverage"/device:GPU:1*
T0*
_output_shapes

: : *
_class
loc:@dense/bias*
N
?
*dense/bias/ExponentialMovingAverage/AssignAssign#dense/bias/ExponentialMovingAverage1cond_17/Merge_dense/bias/ExponentialMovingAverage"/device:GPU:1*
_class
loc:@dense/bias*
T0*
_output_shapes
: 
?
(dense/bias/ExponentialMovingAverage/readIdentity#dense/bias/ExponentialMovingAverage"/device:GPU:1*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
?
IsVariableInitialized_18IsVariableInitializedFCU_muiltDense_x0/dense/kernel"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
dtype0*
_output_shapes
: 
~
cond_18/SwitchSwitchIsVariableInitialized_18IsVariableInitialized_18"/device:CPU:0*
_output_shapes
: : *
T0

^
cond_18/switch_tIdentitycond_18/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

\
cond_18/switch_fIdentitycond_18/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
e
cond_18/pred_idIdentityIsVariableInitialized_18"/device:CPU:0*
_output_shapes
: *
T0

g
cond_18/readIdentitycond_18/read/Switch:1"/device:CPU:0*
T0*
_output_shapes

:  
?
cond_18/read/Switch	RefSwitchFCU_muiltDense_x0/dense/kernelcond_18/pred_id"/device:GPU:1*(
_output_shapes
:  :  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
T0
?
cond_18/Switch_1Switch9FCU_muiltDense_x0/dense/kernel/Initializer/random_uniformcond_18/pred_id*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*(
_output_shapes
:  :  *
T0
y
cond_18/MergeMergecond_18/Switch_1cond_18/read"/device:CPU:0* 
_output_shapes
:  : *
N*
T0
?
7FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage
VariableV2"/device:GPU:1*
dtype0*
_output_shapes

:  *
shape
:  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
MFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedFCU_muiltDense_x0/dense/kernel"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
dtype0*
_output_shapes
: 
?
CFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/SwitchSwitchMFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/IsVariableInitializedMFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: : *
T0
*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
EFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/switch_tIdentityEFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
T0
*
_output_shapes
: *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
EFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/switch_fIdentityCFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/Switch"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
T0
*
_output_shapes
: 
?
DFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/pred_idIdentityMFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_output_shapes
: *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
AFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/readIdentityJFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
_output_shapes

:  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
T0
?
HFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/read/Switch	RefSwitchFCU_muiltDense_x0/dense/kernelDFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*(
_output_shapes
:  :  *
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
EFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/Switch_1Switch9FCU_muiltDense_x0/dense/kernel/Initializer/random_uniformDFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/pred_id*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
T0*(
_output_shapes
:  :  
?
BFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/MergeMergeEFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/Switch_1AFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/read"/device:GPU:1*
T0*
N*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel* 
_output_shapes
:  : 
?
Kcond_18/read/Switch_FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverageSwitchBFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/Mergecond_18/pred_id"/device:GPU:1*
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*(
_output_shapes
:  :  
?
Dcond_18/read_FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverageIdentityMcond_18/read/Switch_FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage:1"/device:GPU:1*
T0*
_output_shapes

:  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
Econd_18/Merge_FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverageMergecond_18/Switch_1Dcond_18/read_FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage"/device:GPU:1* 
_output_shapes
:  : *
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
N
?
>FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/AssignAssign7FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverageEcond_18/Merge_FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage"/device:GPU:1*
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes

:  
?
<FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/readIdentity7FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes

:  *
T0
?
IsVariableInitialized_19IsVariableInitializedFCU_muiltDense_x0/dense/bias"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
_output_shapes
: *
dtype0
~
cond_19/SwitchSwitchIsVariableInitialized_19IsVariableInitialized_19"/device:CPU:0*
T0
*
_output_shapes
: : 
^
cond_19/switch_tIdentitycond_19/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

\
cond_19/switch_fIdentitycond_19/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
e
cond_19/pred_idIdentityIsVariableInitialized_19"/device:CPU:0*
T0
*
_output_shapes
: 
c
cond_19/readIdentitycond_19/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
: 
?
cond_19/read/Switch	RefSwitchFCU_muiltDense_x0/dense/biascond_19/pred_id"/device:GPU:1*
T0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias* 
_output_shapes
: : 
?
cond_19/Switch_1Switch.FCU_muiltDense_x0/dense/bias/Initializer/zeroscond_19/pred_id* 
_output_shapes
: : */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0
u
cond_19/MergeMergecond_19/Switch_1cond_19/read"/device:CPU:0*
_output_shapes

: : *
T0*
N
?
5FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage
VariableV2"/device:GPU:1*
shape: *
dtype0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
_output_shapes
: 
?
KFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedFCU_muiltDense_x0/dense/bias"/device:GPU:1*
_output_shapes
: *
dtype0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
AFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/SwitchSwitchKFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/IsVariableInitializedKFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
_output_shapes
: : *
T0

?
CFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/switch_tIdentityCFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0

?
CFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/switch_fIdentityAFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/Switch"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0
*
_output_shapes
: 
?
BFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/pred_idIdentityKFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: *
T0
*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
?FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/readIdentityHFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
_output_shapes
: *
T0
?
FFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/read/Switch	RefSwitchFCU_muiltDense_x0/dense/biasBFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/pred_id"/device:GPU:1* 
_output_shapes
: : *
T0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
CFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/Switch_1Switch.FCU_muiltDense_x0/dense/bias/Initializer/zerosBFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/pred_id* 
_output_shapes
: : *
T0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
@FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/MergeMergeCFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/Switch_1?FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/read"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0*
_output_shapes

: : *
N
?
Icond_19/read/Switch_FCU_muiltDense_x0/dense/bias/ExponentialMovingAverageSwitch@FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/Mergecond_19/pred_id"/device:GPU:1* 
_output_shapes
: : */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0
?
Bcond_19/read_FCU_muiltDense_x0/dense/bias/ExponentialMovingAverageIdentityKcond_19/read/Switch_FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage:1"/device:GPU:1*
T0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
_output_shapes
: 
?
Ccond_19/Merge_FCU_muiltDense_x0/dense/bias/ExponentialMovingAverageMergecond_19/Switch_1Bcond_19/read_FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage"/device:GPU:1*
N*
T0*
_output_shapes

: : */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
<FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/AssignAssign5FCU_muiltDense_x0/dense/bias/ExponentialMovingAverageCcond_19/Merge_FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage"/device:GPU:1*
_output_shapes
: *
T0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
:FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/readIdentity5FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage"/device:GPU:1*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0
?
IsVariableInitialized_20IsVariableInitializedFCU_muiltDense_x0/beta"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta*
dtype0*
_output_shapes
: 
~
cond_20/SwitchSwitchIsVariableInitialized_20IsVariableInitialized_20"/device:CPU:0*
T0
*
_output_shapes
: : 
^
cond_20/switch_tIdentitycond_20/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

\
cond_20/switch_fIdentitycond_20/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
e
cond_20/pred_idIdentityIsVariableInitialized_20"/device:CPU:0*
_output_shapes
: *
T0

c
cond_20/readIdentitycond_20/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
: 
?
cond_20/read/Switch	RefSwitchFCU_muiltDense_x0/betacond_20/pred_id"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta*
T0* 
_output_shapes
: : 
?
cond_20/Switch_1Switch(FCU_muiltDense_x0/beta/Initializer/zeroscond_20/pred_id*
T0*)
_class
loc:@FCU_muiltDense_x0/beta* 
_output_shapes
: : 
u
cond_20/MergeMergecond_20/Switch_1cond_20/read"/device:CPU:0*
N*
T0*
_output_shapes

: : 
?
/FCU_muiltDense_x0/beta/ExponentialMovingAverage
VariableV2"/device:GPU:1*
dtype0*)
_class
loc:@FCU_muiltDense_x0/beta*
shape: *
_output_shapes
: 
?
EFCU_muiltDense_x0/beta/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedFCU_muiltDense_x0/beta"/device:GPU:1*
_output_shapes
: *
dtype0*)
_class
loc:@FCU_muiltDense_x0/beta
?
;FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/SwitchSwitchEFCU_muiltDense_x0/beta/ExponentialMovingAverage/IsVariableInitializedEFCU_muiltDense_x0/beta/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta*
T0
*
_output_shapes
: : 
?
=FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/switch_tIdentity=FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
_output_shapes
: *)
_class
loc:@FCU_muiltDense_x0/beta*
T0

?
=FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/switch_fIdentity;FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/Switch"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta*
_output_shapes
: *
T0

?
<FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/pred_idIdentityEFCU_muiltDense_x0/beta/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: *
T0
*)
_class
loc:@FCU_muiltDense_x0/beta
?
9FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/readIdentityBFCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
T0*
_output_shapes
: *)
_class
loc:@FCU_muiltDense_x0/beta
?
@FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/read/Switch	RefSwitchFCU_muiltDense_x0/beta<FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta*
T0* 
_output_shapes
: : 
?
=FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/Switch_1Switch(FCU_muiltDense_x0/beta/Initializer/zeros<FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/pred_id*
T0* 
_output_shapes
: : *)
_class
loc:@FCU_muiltDense_x0/beta
?
:FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/MergeMerge=FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/Switch_19FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/read"/device:GPU:1*
N*
T0*
_output_shapes

: : *)
_class
loc:@FCU_muiltDense_x0/beta
?
Ccond_20/read/Switch_FCU_muiltDense_x0/beta/ExponentialMovingAverageSwitch:FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/Mergecond_20/pred_id"/device:GPU:1* 
_output_shapes
: : *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
<cond_20/read_FCU_muiltDense_x0/beta/ExponentialMovingAverageIdentityEcond_20/read/Switch_FCU_muiltDense_x0/beta/ExponentialMovingAverage:1"/device:GPU:1*
_output_shapes
: *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
=cond_20/Merge_FCU_muiltDense_x0/beta/ExponentialMovingAverageMergecond_20/Switch_1<cond_20/read_FCU_muiltDense_x0/beta/ExponentialMovingAverage"/device:GPU:1*
_output_shapes

: : *
N*
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
6FCU_muiltDense_x0/beta/ExponentialMovingAverage/AssignAssign/FCU_muiltDense_x0/beta/ExponentialMovingAverage=cond_20/Merge_FCU_muiltDense_x0/beta/ExponentialMovingAverage"/device:GPU:1*
_output_shapes
: *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
4FCU_muiltDense_x0/beta/ExponentialMovingAverage/readIdentity/FCU_muiltDense_x0/beta/ExponentialMovingAverage"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta*
_output_shapes
: *
T0
?
IsVariableInitialized_21IsVariableInitializedFCU_muiltDense_x0/gamma"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: *
dtype0
~
cond_21/SwitchSwitchIsVariableInitialized_21IsVariableInitialized_21"/device:CPU:0*
_output_shapes
: : *
T0

^
cond_21/switch_tIdentitycond_21/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
\
cond_21/switch_fIdentitycond_21/Switch"/device:CPU:0*
_output_shapes
: *
T0

e
cond_21/pred_idIdentityIsVariableInitialized_21"/device:CPU:0*
T0
*
_output_shapes
: 
c
cond_21/readIdentitycond_21/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
: 
?
cond_21/read/Switch	RefSwitchFCU_muiltDense_x0/gammacond_21/pred_id"/device:GPU:1* 
_output_shapes
: : *
T0**
_class 
loc:@FCU_muiltDense_x0/gamma
?
cond_21/Switch_1Switch(FCU_muiltDense_x0/gamma/Initializer/onescond_21/pred_id* 
_output_shapes
: : **
_class 
loc:@FCU_muiltDense_x0/gamma*
T0
u
cond_21/MergeMergecond_21/Switch_1cond_21/read"/device:CPU:0*
T0*
_output_shapes

: : *
N
?
0FCU_muiltDense_x0/gamma/ExponentialMovingAverage
VariableV2"/device:GPU:1*
_output_shapes
: *
dtype0**
_class 
loc:@FCU_muiltDense_x0/gamma*
shape: 
?
FFCU_muiltDense_x0/gamma/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedFCU_muiltDense_x0/gamma"/device:GPU:1*
_output_shapes
: *
dtype0**
_class 
loc:@FCU_muiltDense_x0/gamma
?
<FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/SwitchSwitchFFCU_muiltDense_x0/gamma/ExponentialMovingAverage/IsVariableInitializedFFCU_muiltDense_x0/gamma/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: : *
T0
**
_class 
loc:@FCU_muiltDense_x0/gamma
?
>FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/switch_tIdentity>FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*
_output_shapes
: **
_class 
loc:@FCU_muiltDense_x0/gamma*
T0

?
>FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/switch_fIdentity<FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
_output_shapes
: *
T0
**
_class 
loc:@FCU_muiltDense_x0/gamma
?
=FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/pred_idIdentityFFCU_muiltDense_x0/gamma/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: *
T0

?
:FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/readIdentityCFCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
T0**
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: 
?
AFCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/read/Switch	RefSwitchFCU_muiltDense_x0/gamma=FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*
T0* 
_output_shapes
: : **
_class 
loc:@FCU_muiltDense_x0/gamma
?
>FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/Switch_1Switch(FCU_muiltDense_x0/gamma/Initializer/ones=FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/pred_id*
T0* 
_output_shapes
: : **
_class 
loc:@FCU_muiltDense_x0/gamma
?
;FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/MergeMerge>FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/Switch_1:FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/read"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
T0*
N*
_output_shapes

: : 
?
Dcond_21/read/Switch_FCU_muiltDense_x0/gamma/ExponentialMovingAverageSwitch;FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/Mergecond_21/pred_id"/device:GPU:1*
T0* 
_output_shapes
: : **
_class 
loc:@FCU_muiltDense_x0/gamma
?
=cond_21/read_FCU_muiltDense_x0/gamma/ExponentialMovingAverageIdentityFcond_21/read/Switch_FCU_muiltDense_x0/gamma/ExponentialMovingAverage:1"/device:GPU:1*
_output_shapes
: **
_class 
loc:@FCU_muiltDense_x0/gamma*
T0
?
>cond_21/Merge_FCU_muiltDense_x0/gamma/ExponentialMovingAverageMergecond_21/Switch_1=cond_21/read_FCU_muiltDense_x0/gamma/ExponentialMovingAverage"/device:GPU:1*
T0*
N*
_output_shapes

: : **
_class 
loc:@FCU_muiltDense_x0/gamma
?
7FCU_muiltDense_x0/gamma/ExponentialMovingAverage/AssignAssign0FCU_muiltDense_x0/gamma/ExponentialMovingAverage>cond_21/Merge_FCU_muiltDense_x0/gamma/ExponentialMovingAverage"/device:GPU:1*
T0**
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: 
?
5FCU_muiltDense_x0/gamma/ExponentialMovingAverage/readIdentity0FCU_muiltDense_x0/gamma/ExponentialMovingAverage"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
T0*
_output_shapes
: 
?
IsVariableInitialized_22IsVariableInitializedOutput_/dense/kernel"/device:GPU:1*
_output_shapes
: *'
_class
loc:@Output_/dense/kernel*
dtype0
~
cond_22/SwitchSwitchIsVariableInitialized_22IsVariableInitialized_22"/device:CPU:0*
_output_shapes
: : *
T0

^
cond_22/switch_tIdentitycond_22/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
\
cond_22/switch_fIdentitycond_22/Switch"/device:CPU:0*
_output_shapes
: *
T0

e
cond_22/pred_idIdentityIsVariableInitialized_22"/device:CPU:0*
T0
*
_output_shapes
: 
g
cond_22/readIdentitycond_22/read/Switch:1"/device:CPU:0*
_output_shapes

: *
T0
?
cond_22/read/Switch	RefSwitchOutput_/dense/kernelcond_22/pred_id"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*(
_output_shapes
: : *
T0
?
cond_22/Switch_1Switch/Output_/dense/kernel/Initializer/random_uniformcond_22/pred_id*(
_output_shapes
: : *'
_class
loc:@Output_/dense/kernel*
T0
y
cond_22/MergeMergecond_22/Switch_1cond_22/read"/device:CPU:0*
T0* 
_output_shapes
: : *
N
?
-Output_/dense/kernel/ExponentialMovingAverage
VariableV2"/device:GPU:1*
dtype0*'
_class
loc:@Output_/dense/kernel*
shape
: *
_output_shapes

: 
?
COutput_/dense/kernel/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedOutput_/dense/kernel"/device:GPU:1*
dtype0*
_output_shapes
: *'
_class
loc:@Output_/dense/kernel
?
9Output_/dense/kernel/ExponentialMovingAverage/cond/SwitchSwitchCOutput_/dense/kernel/ExponentialMovingAverage/IsVariableInitializedCOutput_/dense/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: : *
T0
*'
_class
loc:@Output_/dense/kernel
?
;Output_/dense/kernel/ExponentialMovingAverage/cond/switch_tIdentity;Output_/dense/kernel/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*
_output_shapes
: *
T0

?
;Output_/dense/kernel/ExponentialMovingAverage/cond/switch_fIdentity9Output_/dense/kernel/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
T0
*'
_class
loc:@Output_/dense/kernel*
_output_shapes
: 
?
:Output_/dense/kernel/ExponentialMovingAverage/cond/pred_idIdentityCOutput_/dense/kernel/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
T0
*
_output_shapes
: *'
_class
loc:@Output_/dense/kernel
?
7Output_/dense/kernel/ExponentialMovingAverage/cond/readIdentity@Output_/dense/kernel/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*
_output_shapes

: *
T0*'
_class
loc:@Output_/dense/kernel
?
>Output_/dense/kernel/ExponentialMovingAverage/cond/read/Switch	RefSwitchOutput_/dense/kernel:Output_/dense/kernel/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*
T0*(
_output_shapes
: : 
?
;Output_/dense/kernel/ExponentialMovingAverage/cond/Switch_1Switch/Output_/dense/kernel/Initializer/random_uniform:Output_/dense/kernel/ExponentialMovingAverage/cond/pred_id*'
_class
loc:@Output_/dense/kernel*(
_output_shapes
: : *
T0
?
8Output_/dense/kernel/ExponentialMovingAverage/cond/MergeMerge;Output_/dense/kernel/ExponentialMovingAverage/cond/Switch_17Output_/dense/kernel/ExponentialMovingAverage/cond/read"/device:GPU:1*
T0* 
_output_shapes
: : *
N*'
_class
loc:@Output_/dense/kernel
?
Acond_22/read/Switch_Output_/dense/kernel/ExponentialMovingAverageSwitch8Output_/dense/kernel/ExponentialMovingAverage/cond/Mergecond_22/pred_id"/device:GPU:1*(
_output_shapes
: : *
T0*'
_class
loc:@Output_/dense/kernel
?
:cond_22/read_Output_/dense/kernel/ExponentialMovingAverageIdentityCcond_22/read/Switch_Output_/dense/kernel/ExponentialMovingAverage:1"/device:GPU:1*
T0*'
_class
loc:@Output_/dense/kernel*
_output_shapes

: 
?
;cond_22/Merge_Output_/dense/kernel/ExponentialMovingAverageMergecond_22/Switch_1:cond_22/read_Output_/dense/kernel/ExponentialMovingAverage"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*
T0* 
_output_shapes
: : *
N
?
4Output_/dense/kernel/ExponentialMovingAverage/AssignAssign-Output_/dense/kernel/ExponentialMovingAverage;cond_22/Merge_Output_/dense/kernel/ExponentialMovingAverage"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*
_output_shapes

: *
T0
?
2Output_/dense/kernel/ExponentialMovingAverage/readIdentity-Output_/dense/kernel/ExponentialMovingAverage"/device:GPU:1*
_output_shapes

: *'
_class
loc:@Output_/dense/kernel*
T0
?
IsVariableInitialized_23IsVariableInitializedOutput_/dense/bias"/device:GPU:1*
dtype0*%
_class
loc:@Output_/dense/bias*
_output_shapes
: 
~
cond_23/SwitchSwitchIsVariableInitialized_23IsVariableInitialized_23"/device:CPU:0*
T0
*
_output_shapes
: : 
^
cond_23/switch_tIdentitycond_23/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
\
cond_23/switch_fIdentitycond_23/Switch"/device:CPU:0*
_output_shapes
: *
T0

e
cond_23/pred_idIdentityIsVariableInitialized_23"/device:CPU:0*
_output_shapes
: *
T0

c
cond_23/readIdentitycond_23/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
:
?
cond_23/read/Switch	RefSwitchOutput_/dense/biascond_23/pred_id"/device:GPU:1* 
_output_shapes
::*%
_class
loc:@Output_/dense/bias*
T0
?
cond_23/Switch_1Switch$Output_/dense/bias/Initializer/zeroscond_23/pred_id*%
_class
loc:@Output_/dense/bias*
T0* 
_output_shapes
::
u
cond_23/MergeMergecond_23/Switch_1cond_23/read"/device:CPU:0*
N*
T0*
_output_shapes

:: 
?
+Output_/dense/bias/ExponentialMovingAverage
VariableV2"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
shape:*
dtype0*
_output_shapes
:
?
AOutput_/dense/bias/ExponentialMovingAverage/IsVariableInitializedIsVariableInitializedOutput_/dense/bias"/device:GPU:1*
dtype0*%
_class
loc:@Output_/dense/bias*
_output_shapes
: 
?
7Output_/dense/bias/ExponentialMovingAverage/cond/SwitchSwitchAOutput_/dense/bias/ExponentialMovingAverage/IsVariableInitializedAOutput_/dense/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
T0
*
_output_shapes
: : 
?
9Output_/dense/bias/ExponentialMovingAverage/cond/switch_tIdentity9Output_/dense/bias/ExponentialMovingAverage/cond/Switch:1"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
T0
*
_output_shapes
: 
?
9Output_/dense/bias/ExponentialMovingAverage/cond/switch_fIdentity7Output_/dense/bias/ExponentialMovingAverage/cond/Switch"/device:GPU:1*
T0
*
_output_shapes
: *%
_class
loc:@Output_/dense/bias
?
8Output_/dense/bias/ExponentialMovingAverage/cond/pred_idIdentityAOutput_/dense/bias/ExponentialMovingAverage/IsVariableInitialized"/device:GPU:1*
_output_shapes
: *%
_class
loc:@Output_/dense/bias*
T0

?
5Output_/dense/bias/ExponentialMovingAverage/cond/readIdentity>Output_/dense/bias/ExponentialMovingAverage/cond/read/Switch:1"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
T0*
_output_shapes
:
?
<Output_/dense/bias/ExponentialMovingAverage/cond/read/Switch	RefSwitchOutput_/dense/bias8Output_/dense/bias/ExponentialMovingAverage/cond/pred_id"/device:GPU:1*
T0* 
_output_shapes
::*%
_class
loc:@Output_/dense/bias
?
9Output_/dense/bias/ExponentialMovingAverage/cond/Switch_1Switch$Output_/dense/bias/Initializer/zeros8Output_/dense/bias/ExponentialMovingAverage/cond/pred_id* 
_output_shapes
::*%
_class
loc:@Output_/dense/bias*
T0
?
6Output_/dense/bias/ExponentialMovingAverage/cond/MergeMerge9Output_/dense/bias/ExponentialMovingAverage/cond/Switch_15Output_/dense/bias/ExponentialMovingAverage/cond/read"/device:GPU:1*
_output_shapes

:: *
N*
T0*%
_class
loc:@Output_/dense/bias
?
?cond_23/read/Switch_Output_/dense/bias/ExponentialMovingAverageSwitch6Output_/dense/bias/ExponentialMovingAverage/cond/Mergecond_23/pred_id"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
T0* 
_output_shapes
::
?
8cond_23/read_Output_/dense/bias/ExponentialMovingAverageIdentityAcond_23/read/Switch_Output_/dense/bias/ExponentialMovingAverage:1"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
_output_shapes
:*
T0
?
9cond_23/Merge_Output_/dense/bias/ExponentialMovingAverageMergecond_23/Switch_18cond_23/read_Output_/dense/bias/ExponentialMovingAverage"/device:GPU:1*
N*
_output_shapes

:: *%
_class
loc:@Output_/dense/bias*
T0
?
2Output_/dense/bias/ExponentialMovingAverage/AssignAssign+Output_/dense/bias/ExponentialMovingAverage9cond_23/Merge_Output_/dense/bias/ExponentialMovingAverage"/device:GPU:1*
_output_shapes
:*%
_class
loc:@Output_/dense/bias*
T0
?
0Output_/dense/bias/ExponentialMovingAverage/readIdentity+Output_/dense/bias/ExponentialMovingAverage"/device:GPU:1*
T0*%
_class
loc:@Output_/dense/bias*
_output_shapes
:
r
ExponentialMovingAverage/decayConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *w??
r
ExponentialMovingAverage/add/xConst"/device:CPU:0*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
ExponentialMovingAverage/addAddV2ExponentialMovingAverage/add/xPlaceholder"/device:CPU:0*
T0*
_output_shapes
:
t
 ExponentialMovingAverage/add_1/xConst"/device:CPU:0*
_output_shapes
: *
valueB
 *   A*
dtype0
?
ExponentialMovingAverage/add_1AddV2 ExponentialMovingAverage/add_1/xPlaceholder"/device:CPU:0*
_output_shapes
:*
T0
?
 ExponentialMovingAverage/truedivRealDivExponentialMovingAverage/addExponentialMovingAverage/add_1"/device:CPU:0*
T0*
_output_shapes
:
?
 ExponentialMovingAverage/MinimumMinimumExponentialMovingAverage/decay ExponentialMovingAverage/truediv"/device:CPU:0*
_output_shapes
:*
T0
?
.ExponentialMovingAverage/AssignMovingAvg/sub/xConst"/device:CPU:0*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
,ExponentialMovingAverage/AssignMovingAvg/subSub.ExponentialMovingAverage/AssignMovingAvg/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
T0*
_output_shapes
:
?
.ExponentialMovingAverage/AssignMovingAvg/sub_1Sub9CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/read CCN_1Conv_x0/convA10/kernel/read"/device:CPU:0*
T0*#
_output_shapes
:?
?
,ExponentialMovingAverage/AssignMovingAvg/mulMul.ExponentialMovingAverage/AssignMovingAvg/sub_1,ExponentialMovingAverage/AssignMovingAvg/sub"/device:CPU:0*
T0*
_output_shapes
:
?
(ExponentialMovingAverage/AssignMovingAvg	AssignSub4CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage,ExponentialMovingAverage/AssignMovingAvg/mul"/device:GPU:1*
T0*#
_output_shapes
:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
0ExponentialMovingAverage/AssignMovingAvg_1/sub/xConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
.ExponentialMovingAverage/AssignMovingAvg_1/subSub0ExponentialMovingAverage/AssignMovingAvg_1/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
T0*
_output_shapes
:
?
0ExponentialMovingAverage/AssignMovingAvg_1/sub_1Sub7CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/readCCN_1Conv_x0/convA10/bias/read"/device:CPU:0*
T0*
_output_shapes	
:?
?
.ExponentialMovingAverage/AssignMovingAvg_1/mulMul0ExponentialMovingAverage/AssignMovingAvg_1/sub_1.ExponentialMovingAverage/AssignMovingAvg_1/sub"/device:CPU:0*
_output_shapes
:*
T0
?
*ExponentialMovingAverage/AssignMovingAvg_1	AssignSub2CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage.ExponentialMovingAverage/AssignMovingAvg_1/mul"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0*
_output_shapes	
:?
?
0ExponentialMovingAverage/AssignMovingAvg_2/sub/xConst"/device:CPU:0*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
.ExponentialMovingAverage/AssignMovingAvg_2/subSub0ExponentialMovingAverage/AssignMovingAvg_2/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
0ExponentialMovingAverage/AssignMovingAvg_2/sub_1Sub9CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/read CCN_1Conv_x0/convB10/kernel/read"/device:CPU:0*$
_output_shapes
:??*
T0
?
.ExponentialMovingAverage/AssignMovingAvg_2/mulMul0ExponentialMovingAverage/AssignMovingAvg_2/sub_1.ExponentialMovingAverage/AssignMovingAvg_2/sub"/device:CPU:0*
_output_shapes
:*
T0
?
*ExponentialMovingAverage/AssignMovingAvg_2	AssignSub4CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage.ExponentialMovingAverage/AssignMovingAvg_2/mul"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel
?
0ExponentialMovingAverage/AssignMovingAvg_3/sub/xConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
.ExponentialMovingAverage/AssignMovingAvg_3/subSub0ExponentialMovingAverage/AssignMovingAvg_3/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
0ExponentialMovingAverage/AssignMovingAvg_3/sub_1Sub7CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/readCCN_1Conv_x0/convB10/bias/read"/device:CPU:0*
T0*
_output_shapes	
:?
?
.ExponentialMovingAverage/AssignMovingAvg_3/mulMul0ExponentialMovingAverage/AssignMovingAvg_3/sub_1.ExponentialMovingAverage/AssignMovingAvg_3/sub"/device:CPU:0*
_output_shapes
:*
T0
?
*ExponentialMovingAverage/AssignMovingAvg_3	AssignSub2CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage.ExponentialMovingAverage/AssignMovingAvg_3/mul"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
_output_shapes	
:?*
T0
?
0ExponentialMovingAverage/AssignMovingAvg_4/sub/xConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
.ExponentialMovingAverage/AssignMovingAvg_4/subSub0ExponentialMovingAverage/AssignMovingAvg_4/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
0ExponentialMovingAverage/AssignMovingAvg_4/sub_1Sub9CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/read CCN_1Conv_x0/convB20/kernel/read"/device:CPU:0*$
_output_shapes
:??*
T0
?
.ExponentialMovingAverage/AssignMovingAvg_4/mulMul0ExponentialMovingAverage/AssignMovingAvg_4/sub_1.ExponentialMovingAverage/AssignMovingAvg_4/sub"/device:CPU:0*
_output_shapes
:*
T0
?
*ExponentialMovingAverage/AssignMovingAvg_4	AssignSub4CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage.ExponentialMovingAverage/AssignMovingAvg_4/mul"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
0ExponentialMovingAverage/AssignMovingAvg_5/sub/xConst"/device:CPU:0*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
.ExponentialMovingAverage/AssignMovingAvg_5/subSub0ExponentialMovingAverage/AssignMovingAvg_5/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
0ExponentialMovingAverage/AssignMovingAvg_5/sub_1Sub7CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/readCCN_1Conv_x0/convB20/bias/read"/device:CPU:0*
_output_shapes	
:?*
T0
?
.ExponentialMovingAverage/AssignMovingAvg_5/mulMul0ExponentialMovingAverage/AssignMovingAvg_5/sub_1.ExponentialMovingAverage/AssignMovingAvg_5/sub"/device:CPU:0*
_output_shapes
:*
T0
?
*ExponentialMovingAverage/AssignMovingAvg_5	AssignSub2CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage.ExponentialMovingAverage/AssignMovingAvg_5/mul"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
_output_shapes	
:?
?
0ExponentialMovingAverage/AssignMovingAvg_6/sub/xConst"/device:CPU:0*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
.ExponentialMovingAverage/AssignMovingAvg_6/subSub0ExponentialMovingAverage/AssignMovingAvg_6/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
0ExponentialMovingAverage/AssignMovingAvg_6/sub_1Sub9CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/read CCN_1Conv_x0/convA11/kernel/read"/device:CPU:0*
T0*$
_output_shapes
:??
?
.ExponentialMovingAverage/AssignMovingAvg_6/mulMul0ExponentialMovingAverage/AssignMovingAvg_6/sub_1.ExponentialMovingAverage/AssignMovingAvg_6/sub"/device:CPU:0*
_output_shapes
:*
T0
?
*ExponentialMovingAverage/AssignMovingAvg_6	AssignSub4CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage.ExponentialMovingAverage/AssignMovingAvg_6/mul"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
0ExponentialMovingAverage/AssignMovingAvg_7/sub/xConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
.ExponentialMovingAverage/AssignMovingAvg_7/subSub0ExponentialMovingAverage/AssignMovingAvg_7/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
T0*
_output_shapes
:
?
0ExponentialMovingAverage/AssignMovingAvg_7/sub_1Sub7CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/readCCN_1Conv_x0/convA11/bias/read"/device:CPU:0*
_output_shapes	
:?*
T0
?
.ExponentialMovingAverage/AssignMovingAvg_7/mulMul0ExponentialMovingAverage/AssignMovingAvg_7/sub_1.ExponentialMovingAverage/AssignMovingAvg_7/sub"/device:CPU:0*
T0*
_output_shapes
:
?
*ExponentialMovingAverage/AssignMovingAvg_7	AssignSub2CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage.ExponentialMovingAverage/AssignMovingAvg_7/mul"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?
?
0ExponentialMovingAverage/AssignMovingAvg_8/sub/xConst"/device:CPU:0*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
.ExponentialMovingAverage/AssignMovingAvg_8/subSub0ExponentialMovingAverage/AssignMovingAvg_8/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
0ExponentialMovingAverage/AssignMovingAvg_8/sub_1Sub9CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/read CCN_1Conv_x0/convB11/kernel/read"/device:CPU:0*
T0*$
_output_shapes
:??
?
.ExponentialMovingAverage/AssignMovingAvg_8/mulMul0ExponentialMovingAverage/AssignMovingAvg_8/sub_1.ExponentialMovingAverage/AssignMovingAvg_8/sub"/device:CPU:0*
_output_shapes
:*
T0
?
*ExponentialMovingAverage/AssignMovingAvg_8	AssignSub4CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage.ExponentialMovingAverage/AssignMovingAvg_8/mul"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
0ExponentialMovingAverage/AssignMovingAvg_9/sub/xConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
.ExponentialMovingAverage/AssignMovingAvg_9/subSub0ExponentialMovingAverage/AssignMovingAvg_9/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
0ExponentialMovingAverage/AssignMovingAvg_9/sub_1Sub7CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/readCCN_1Conv_x0/convB11/bias/read"/device:CPU:0*
_output_shapes	
:?*
T0
?
.ExponentialMovingAverage/AssignMovingAvg_9/mulMul0ExponentialMovingAverage/AssignMovingAvg_9/sub_1.ExponentialMovingAverage/AssignMovingAvg_9/sub"/device:CPU:0*
_output_shapes
:*
T0
?
*ExponentialMovingAverage/AssignMovingAvg_9	AssignSub2CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage.ExponentialMovingAverage/AssignMovingAvg_9/mul"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?
?
1ExponentialMovingAverage/AssignMovingAvg_10/sub/xConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
/ExponentialMovingAverage/AssignMovingAvg_10/subSub1ExponentialMovingAverage/AssignMovingAvg_10/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
1ExponentialMovingAverage/AssignMovingAvg_10/sub_1Sub9CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/read CCN_1Conv_x0/convB21/kernel/read"/device:CPU:0*$
_output_shapes
:??*
T0
?
/ExponentialMovingAverage/AssignMovingAvg_10/mulMul1ExponentialMovingAverage/AssignMovingAvg_10/sub_1/ExponentialMovingAverage/AssignMovingAvg_10/sub"/device:CPU:0*
_output_shapes
:*
T0
?
+ExponentialMovingAverage/AssignMovingAvg_10	AssignSub4CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_10/mul"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
1ExponentialMovingAverage/AssignMovingAvg_11/sub/xConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
/ExponentialMovingAverage/AssignMovingAvg_11/subSub1ExponentialMovingAverage/AssignMovingAvg_11/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
T0*
_output_shapes
:
?
1ExponentialMovingAverage/AssignMovingAvg_11/sub_1Sub7CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/readCCN_1Conv_x0/convB21/bias/read"/device:CPU:0*
_output_shapes	
:?*
T0
?
/ExponentialMovingAverage/AssignMovingAvg_11/mulMul1ExponentialMovingAverage/AssignMovingAvg_11/sub_1/ExponentialMovingAverage/AssignMovingAvg_11/sub"/device:CPU:0*
T0*
_output_shapes
:
?
+ExponentialMovingAverage/AssignMovingAvg_11	AssignSub2CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_11/mul"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0*
_output_shapes	
:?
?
1ExponentialMovingAverage/AssignMovingAvg_12/sub/xConst"/device:CPU:0*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
/ExponentialMovingAverage/AssignMovingAvg_12/subSub1ExponentialMovingAverage/AssignMovingAvg_12/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
T0*
_output_shapes
:
?
1ExponentialMovingAverage/AssignMovingAvg_12/sub_1Sub-Conv_out__/beta/ExponentialMovingAverage/readConv_out__/beta/read"/device:CPU:0*
_output_shapes	
:?*
T0
?
/ExponentialMovingAverage/AssignMovingAvg_12/mulMul1ExponentialMovingAverage/AssignMovingAvg_12/sub_1/ExponentialMovingAverage/AssignMovingAvg_12/sub"/device:CPU:0*
_output_shapes
:*
T0
?
+ExponentialMovingAverage/AssignMovingAvg_12	AssignSub(Conv_out__/beta/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_12/mul"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
_output_shapes	
:?*
T0
?
1ExponentialMovingAverage/AssignMovingAvg_13/sub/xConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
/ExponentialMovingAverage/AssignMovingAvg_13/subSub1ExponentialMovingAverage/AssignMovingAvg_13/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
1ExponentialMovingAverage/AssignMovingAvg_13/sub_1Sub.Conv_out__/gamma/ExponentialMovingAverage/readConv_out__/gamma/read"/device:CPU:0*
T0*
_output_shapes	
:?
?
/ExponentialMovingAverage/AssignMovingAvg_13/mulMul1ExponentialMovingAverage/AssignMovingAvg_13/sub_1/ExponentialMovingAverage/AssignMovingAvg_13/sub"/device:CPU:0*
T0*
_output_shapes
:
?
+ExponentialMovingAverage/AssignMovingAvg_13	AssignSub)Conv_out__/gamma/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_13/mul"/device:GPU:1*
_output_shapes	
:?*
T0*#
_class
loc:@Conv_out__/gamma
?
1ExponentialMovingAverage/AssignMovingAvg_14/sub/xConst"/device:CPU:0*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
/ExponentialMovingAverage/AssignMovingAvg_14/subSub1ExponentialMovingAverage/AssignMovingAvg_14/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
T0*
_output_shapes
:
?
1ExponentialMovingAverage/AssignMovingAvg_14/sub_1Sub@Reconstruction_Output/dense/kernel/ExponentialMovingAverage/read'Reconstruction_Output/dense/kernel/read"/device:CPU:0*
T0*
_output_shapes
:	?
?
/ExponentialMovingAverage/AssignMovingAvg_14/mulMul1ExponentialMovingAverage/AssignMovingAvg_14/sub_1/ExponentialMovingAverage/AssignMovingAvg_14/sub"/device:CPU:0*
T0*
_output_shapes
:
?
+ExponentialMovingAverage/AssignMovingAvg_14	AssignSub;Reconstruction_Output/dense/kernel/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_14/mul"/device:GPU:1*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?
?
1ExponentialMovingAverage/AssignMovingAvg_15/sub/xConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
/ExponentialMovingAverage/AssignMovingAvg_15/subSub1ExponentialMovingAverage/AssignMovingAvg_15/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
T0*
_output_shapes
:
?
1ExponentialMovingAverage/AssignMovingAvg_15/sub_1Sub>Reconstruction_Output/dense/bias/ExponentialMovingAverage/read%Reconstruction_Output/dense/bias/read"/device:CPU:0*
_output_shapes
:*
T0
?
/ExponentialMovingAverage/AssignMovingAvg_15/mulMul1ExponentialMovingAverage/AssignMovingAvg_15/sub_1/ExponentialMovingAverage/AssignMovingAvg_15/sub"/device:CPU:0*
T0*
_output_shapes
:
?
+ExponentialMovingAverage/AssignMovingAvg_15	AssignSub9Reconstruction_Output/dense/bias/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_15/mul"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
:*
T0
?
1ExponentialMovingAverage/AssignMovingAvg_16/sub/xConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
/ExponentialMovingAverage/AssignMovingAvg_16/subSub1ExponentialMovingAverage/AssignMovingAvg_16/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
T0*
_output_shapes
:
?
1ExponentialMovingAverage/AssignMovingAvg_16/sub_1Sub*dense/kernel/ExponentialMovingAverage/readdense/kernel/read"/device:CPU:0*
_output_shapes
:	? *
T0
?
/ExponentialMovingAverage/AssignMovingAvg_16/mulMul1ExponentialMovingAverage/AssignMovingAvg_16/sub_1/ExponentialMovingAverage/AssignMovingAvg_16/sub"/device:CPU:0*
T0*
_output_shapes
:
?
+ExponentialMovingAverage/AssignMovingAvg_16	AssignSub%dense/kernel/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_16/mul"/device:GPU:1*
T0*
_output_shapes
:	? *
_class
loc:@dense/kernel
?
1ExponentialMovingAverage/AssignMovingAvg_17/sub/xConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
/ExponentialMovingAverage/AssignMovingAvg_17/subSub1ExponentialMovingAverage/AssignMovingAvg_17/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
1ExponentialMovingAverage/AssignMovingAvg_17/sub_1Sub(dense/bias/ExponentialMovingAverage/readdense/bias/read"/device:CPU:0*
_output_shapes
: *
T0
?
/ExponentialMovingAverage/AssignMovingAvg_17/mulMul1ExponentialMovingAverage/AssignMovingAvg_17/sub_1/ExponentialMovingAverage/AssignMovingAvg_17/sub"/device:CPU:0*
T0*
_output_shapes
:
?
+ExponentialMovingAverage/AssignMovingAvg_17	AssignSub#dense/bias/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_17/mul"/device:GPU:1*
_output_shapes
: *
_class
loc:@dense/bias*
T0
?
1ExponentialMovingAverage/AssignMovingAvg_18/sub/xConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
/ExponentialMovingAverage/AssignMovingAvg_18/subSub1ExponentialMovingAverage/AssignMovingAvg_18/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
T0*
_output_shapes
:
?
1ExponentialMovingAverage/AssignMovingAvg_18/sub_1Sub<FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/read#FCU_muiltDense_x0/dense/kernel/read"/device:CPU:0*
T0*
_output_shapes

:  
?
/ExponentialMovingAverage/AssignMovingAvg_18/mulMul1ExponentialMovingAverage/AssignMovingAvg_18/sub_1/ExponentialMovingAverage/AssignMovingAvg_18/sub"/device:CPU:0*
T0*
_output_shapes
:
?
+ExponentialMovingAverage/AssignMovingAvg_18	AssignSub7FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_18/mul"/device:GPU:1*
T0*
_output_shapes

:  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
1ExponentialMovingAverage/AssignMovingAvg_19/sub/xConst"/device:CPU:0*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
/ExponentialMovingAverage/AssignMovingAvg_19/subSub1ExponentialMovingAverage/AssignMovingAvg_19/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
T0*
_output_shapes
:
?
1ExponentialMovingAverage/AssignMovingAvg_19/sub_1Sub:FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/read!FCU_muiltDense_x0/dense/bias/read"/device:CPU:0*
T0*
_output_shapes
: 
?
/ExponentialMovingAverage/AssignMovingAvg_19/mulMul1ExponentialMovingAverage/AssignMovingAvg_19/sub_1/ExponentialMovingAverage/AssignMovingAvg_19/sub"/device:CPU:0*
T0*
_output_shapes
:
?
+ExponentialMovingAverage/AssignMovingAvg_19	AssignSub5FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_19/mul"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0*
_output_shapes
: 
?
1ExponentialMovingAverage/AssignMovingAvg_20/sub/xConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
/ExponentialMovingAverage/AssignMovingAvg_20/subSub1ExponentialMovingAverage/AssignMovingAvg_20/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
1ExponentialMovingAverage/AssignMovingAvg_20/sub_1Sub4FCU_muiltDense_x0/beta/ExponentialMovingAverage/readFCU_muiltDense_x0/beta/read"/device:CPU:0*
_output_shapes
: *
T0
?
/ExponentialMovingAverage/AssignMovingAvg_20/mulMul1ExponentialMovingAverage/AssignMovingAvg_20/sub_1/ExponentialMovingAverage/AssignMovingAvg_20/sub"/device:CPU:0*
_output_shapes
:*
T0
?
+ExponentialMovingAverage/AssignMovingAvg_20	AssignSub/FCU_muiltDense_x0/beta/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_20/mul"/device:GPU:1*
_output_shapes
: *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
1ExponentialMovingAverage/AssignMovingAvg_21/sub/xConst"/device:CPU:0*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
/ExponentialMovingAverage/AssignMovingAvg_21/subSub1ExponentialMovingAverage/AssignMovingAvg_21/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
T0*
_output_shapes
:
?
1ExponentialMovingAverage/AssignMovingAvg_21/sub_1Sub5FCU_muiltDense_x0/gamma/ExponentialMovingAverage/readFCU_muiltDense_x0/gamma/read"/device:CPU:0*
T0*
_output_shapes
: 
?
/ExponentialMovingAverage/AssignMovingAvg_21/mulMul1ExponentialMovingAverage/AssignMovingAvg_21/sub_1/ExponentialMovingAverage/AssignMovingAvg_21/sub"/device:CPU:0*
T0*
_output_shapes
:
?
+ExponentialMovingAverage/AssignMovingAvg_21	AssignSub0FCU_muiltDense_x0/gamma/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_21/mul"/device:GPU:1*
_output_shapes
: *
T0**
_class 
loc:@FCU_muiltDense_x0/gamma
?
1ExponentialMovingAverage/AssignMovingAvg_22/sub/xConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
/ExponentialMovingAverage/AssignMovingAvg_22/subSub1ExponentialMovingAverage/AssignMovingAvg_22/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
1ExponentialMovingAverage/AssignMovingAvg_22/sub_1Sub2Output_/dense/kernel/ExponentialMovingAverage/readOutput_/dense/kernel/read"/device:CPU:0*
_output_shapes

: *
T0
?
/ExponentialMovingAverage/AssignMovingAvg_22/mulMul1ExponentialMovingAverage/AssignMovingAvg_22/sub_1/ExponentialMovingAverage/AssignMovingAvg_22/sub"/device:CPU:0*
_output_shapes
:*
T0
?
+ExponentialMovingAverage/AssignMovingAvg_22	AssignSub-Output_/dense/kernel/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_22/mul"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*
T0*
_output_shapes

: 
?
1ExponentialMovingAverage/AssignMovingAvg_23/sub/xConst"/device:CPU:0*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
/ExponentialMovingAverage/AssignMovingAvg_23/subSub1ExponentialMovingAverage/AssignMovingAvg_23/sub/x ExponentialMovingAverage/Minimum"/device:CPU:0*
_output_shapes
:*
T0
?
1ExponentialMovingAverage/AssignMovingAvg_23/sub_1Sub0Output_/dense/bias/ExponentialMovingAverage/readOutput_/dense/bias/read"/device:CPU:0*
_output_shapes
:*
T0
?
/ExponentialMovingAverage/AssignMovingAvg_23/mulMul1ExponentialMovingAverage/AssignMovingAvg_23/sub_1/ExponentialMovingAverage/AssignMovingAvg_23/sub"/device:CPU:0*
T0*
_output_shapes
:
?
+ExponentialMovingAverage/AssignMovingAvg_23	AssignSub+Output_/dense/bias/ExponentialMovingAverage/ExponentialMovingAverage/AssignMovingAvg_23/mul"/device:GPU:1*
T0*%
_class
loc:@Output_/dense/bias*
_output_shapes
:
?
ExponentialMovingAverageNoOp)^ExponentialMovingAverage/AssignMovingAvg+^ExponentialMovingAverage/AssignMovingAvg_1,^ExponentialMovingAverage/AssignMovingAvg_10,^ExponentialMovingAverage/AssignMovingAvg_11,^ExponentialMovingAverage/AssignMovingAvg_12,^ExponentialMovingAverage/AssignMovingAvg_13,^ExponentialMovingAverage/AssignMovingAvg_14,^ExponentialMovingAverage/AssignMovingAvg_15,^ExponentialMovingAverage/AssignMovingAvg_16,^ExponentialMovingAverage/AssignMovingAvg_17,^ExponentialMovingAverage/AssignMovingAvg_18,^ExponentialMovingAverage/AssignMovingAvg_19+^ExponentialMovingAverage/AssignMovingAvg_2,^ExponentialMovingAverage/AssignMovingAvg_20,^ExponentialMovingAverage/AssignMovingAvg_21,^ExponentialMovingAverage/AssignMovingAvg_22,^ExponentialMovingAverage/AssignMovingAvg_23+^ExponentialMovingAverage/AssignMovingAvg_3+^ExponentialMovingAverage/AssignMovingAvg_4+^ExponentialMovingAverage/AssignMovingAvg_5+^ExponentialMovingAverage/AssignMovingAvg_6+^ExponentialMovingAverage/AssignMovingAvg_7+^ExponentialMovingAverage/AssignMovingAvg_8+^ExponentialMovingAverage/AssignMovingAvg_9"/device:GPU:1
-
group_deps/NoOpNoOp^Adam"/device:CPU:0
C
group_deps/NoOp_1NoOp^ExponentialMovingAverage"/device:GPU:1
G

group_depsNoOp^group_deps/NoOp^group_deps/NoOp_1"/device:CPU:0
?
%BackupVariables/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convA10/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
_output_shapes
: *
dtype0
?
BackupVariables/cond/SwitchSwitch%BackupVariables/IsVariableInitialized%BackupVariables/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: : 
x
BackupVariables/cond/switch_tIdentityBackupVariables/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

v
BackupVariables/cond/switch_fIdentityBackupVariables/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 

BackupVariables/cond/pred_idIdentity%BackupVariables/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond/readIdentity"BackupVariables/cond/read/Switch:1"/device:CPU:0*
T0*#
_output_shapes
:?
?
 BackupVariables/cond/read/Switch	RefSwitchCCN_1Conv_x0/convA10/kernelBackupVariables/cond/pred_id"/device:GPU:1*2
_output_shapes 
:?:?*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
BackupVariables/cond/Switch_1Switch6CCN_1Conv_x0/convA10/kernel/Initializer/random_uniformBackupVariables/cond/pred_id*2
_output_shapes 
:?:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0
?
BackupVariables/cond/MergeMergeBackupVariables/cond/Switch_1BackupVariables/cond/read"/device:CPU:0*
T0*%
_output_shapes
:?: *
N
?
+BackupVariables/CCN_1Conv_x0/convA10/kernel
VariableV2"/device:CPU:0*#
_output_shapes
:?*
shape:?*
dtype0
?
ABackupVariables/CCN_1Conv_x0/convA10/kernel/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convA10/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
_output_shapes
: *
dtype0
?
7BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/SwitchSwitchABackupVariables/CCN_1Conv_x0/convA10/kernel/IsVariableInitializedABackupVariables/CCN_1Conv_x0/convA10/kernel/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: : 
?
9BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/switch_tIdentity9BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
?
9BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/switch_fIdentity7BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
8BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/pred_idIdentityABackupVariables/CCN_1Conv_x0/convA10/kernel/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
5BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/readIdentity>BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/read/Switch:1"/device:CPU:0*#
_output_shapes
:?*
T0
?
<BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/read/Switch	RefSwitchCCN_1Conv_x0/convA10/kernel8BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/pred_id"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*2
_output_shapes 
:?:?
?
9BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/Switch_1Switch6CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform8BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/pred_id*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0*2
_output_shapes 
:?:?
?
6BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/MergeMerge9BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/Switch_15BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/read"/device:CPU:0*
N*%
_output_shapes
:?: *
T0
?
LBackupVariables/cond/read/Switch_BackupVariables/CCN_1Conv_x0/convA10/kernelSwitch6BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/MergeBackupVariables/cond/pred_id"/device:CPU:0*2
_output_shapes 
:?:?*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
EBackupVariables/cond/read_BackupVariables/CCN_1Conv_x0/convA10/kernelIdentityNBackupVariables/cond/read/Switch_BackupVariables/CCN_1Conv_x0/convA10/kernel:1"/device:CPU:0*
T0*#
_output_shapes
:?
?
FBackupVariables/cond/Merge_BackupVariables/CCN_1Conv_x0/convA10/kernelMergeBackupVariables/cond/Switch_1EBackupVariables/cond/read_BackupVariables/CCN_1Conv_x0/convA10/kernel"/device:CPU:0*%
_output_shapes
:?: *
N*
T0
?
2BackupVariables/CCN_1Conv_x0/convA10/kernel/AssignAssign+BackupVariables/CCN_1Conv_x0/convA10/kernelFBackupVariables/cond/Merge_BackupVariables/CCN_1Conv_x0/convA10/kernel"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?*
T0
?
0BackupVariables/CCN_1Conv_x0/convA10/kernel/readIdentity+BackupVariables/CCN_1Conv_x0/convA10/kernel"/device:CPU:0*#
_output_shapes
:?*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convA10/kernel*
T0
?
'BackupVariables/IsVariableInitialized_1IsVariableInitializedCCN_1Conv_x0/convA10/bias"/device:GPU:1*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes
: 
?
BackupVariables/cond_1/SwitchSwitch'BackupVariables/IsVariableInitialized_1'BackupVariables/IsVariableInitialized_1"/device:CPU:0*
T0
*
_output_shapes
: : 
|
BackupVariables/cond_1/switch_tIdentityBackupVariables/cond_1/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

z
BackupVariables/cond_1/switch_fIdentityBackupVariables/cond_1/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_1/pred_idIdentity'BackupVariables/IsVariableInitialized_1"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_1/readIdentity$BackupVariables/cond_1/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
"BackupVariables/cond_1/read/Switch	RefSwitchCCN_1Conv_x0/convA10/biasBackupVariables/cond_1/pred_id"/device:GPU:1*"
_output_shapes
:?:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
BackupVariables/cond_1/Switch_1Switch+CCN_1Conv_x0/convA10/bias/Initializer/zerosBackupVariables/cond_1/pred_id*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0*"
_output_shapes
:?:?
?
BackupVariables/cond_1/MergeMergeBackupVariables/cond_1/Switch_1BackupVariables/cond_1/read"/device:CPU:0*
T0*
N*
_output_shapes
	:?: 
?
)BackupVariables/CCN_1Conv_x0/convA10/bias
VariableV2"/device:CPU:0*
shape:?*
_output_shapes	
:?*
dtype0
?
?BackupVariables/CCN_1Conv_x0/convA10/bias/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convA10/bias"/device:GPU:1*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes
: 
?
5BackupVariables/CCN_1Conv_x0/convA10/bias/cond/SwitchSwitch?BackupVariables/CCN_1Conv_x0/convA10/bias/IsVariableInitialized?BackupVariables/CCN_1Conv_x0/convA10/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
7BackupVariables/CCN_1Conv_x0/convA10/bias/cond/switch_tIdentity7BackupVariables/CCN_1Conv_x0/convA10/bias/cond/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
?
7BackupVariables/CCN_1Conv_x0/convA10/bias/cond/switch_fIdentity5BackupVariables/CCN_1Conv_x0/convA10/bias/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
6BackupVariables/CCN_1Conv_x0/convA10/bias/cond/pred_idIdentity?BackupVariables/CCN_1Conv_x0/convA10/bias/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
3BackupVariables/CCN_1Conv_x0/convA10/bias/cond/readIdentity<BackupVariables/CCN_1Conv_x0/convA10/bias/cond/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
:BackupVariables/CCN_1Conv_x0/convA10/bias/cond/read/Switch	RefSwitchCCN_1Conv_x0/convA10/bias6BackupVariables/CCN_1Conv_x0/convA10/bias/cond/pred_id"/device:GPU:1*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0
?
7BackupVariables/CCN_1Conv_x0/convA10/bias/cond/Switch_1Switch+CCN_1Conv_x0/convA10/bias/Initializer/zeros6BackupVariables/CCN_1Conv_x0/convA10/bias/cond/pred_id*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*"
_output_shapes
:?:?*
T0
?
4BackupVariables/CCN_1Conv_x0/convA10/bias/cond/MergeMerge7BackupVariables/CCN_1Conv_x0/convA10/bias/cond/Switch_13BackupVariables/CCN_1Conv_x0/convA10/bias/cond/read"/device:CPU:0*
_output_shapes
	:?: *
T0*
N
?
LBackupVariables/cond_1/read/Switch_BackupVariables/CCN_1Conv_x0/convA10/biasSwitch4BackupVariables/CCN_1Conv_x0/convA10/bias/cond/MergeBackupVariables/cond_1/pred_id"/device:CPU:0*"
_output_shapes
:?:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
EBackupVariables/cond_1/read_BackupVariables/CCN_1Conv_x0/convA10/biasIdentityNBackupVariables/cond_1/read/Switch_BackupVariables/CCN_1Conv_x0/convA10/bias:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
FBackupVariables/cond_1/Merge_BackupVariables/CCN_1Conv_x0/convA10/biasMergeBackupVariables/cond_1/Switch_1EBackupVariables/cond_1/read_BackupVariables/CCN_1Conv_x0/convA10/bias"/device:CPU:0*
_output_shapes
	:?: *
T0*
N
?
0BackupVariables/CCN_1Conv_x0/convA10/bias/AssignAssign)BackupVariables/CCN_1Conv_x0/convA10/biasFBackupVariables/cond_1/Merge_BackupVariables/CCN_1Conv_x0/convA10/bias"/device:CPU:0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?*
T0
?
.BackupVariables/CCN_1Conv_x0/convA10/bias/readIdentity)BackupVariables/CCN_1Conv_x0/convA10/bias"/device:CPU:0*
_output_shapes	
:?*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convA10/bias
?
'BackupVariables/IsVariableInitialized_2IsVariableInitializedCCN_1Conv_x0/convB10/kernel"/device:GPU:1*
_output_shapes
: *
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel
?
BackupVariables/cond_2/SwitchSwitch'BackupVariables/IsVariableInitialized_2'BackupVariables/IsVariableInitialized_2"/device:CPU:0*
_output_shapes
: : *
T0

|
BackupVariables/cond_2/switch_tIdentityBackupVariables/cond_2/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
z
BackupVariables/cond_2/switch_fIdentityBackupVariables/cond_2/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_2/pred_idIdentity'BackupVariables/IsVariableInitialized_2"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_2/readIdentity$BackupVariables/cond_2/read/Switch:1"/device:CPU:0*
T0*$
_output_shapes
:??
?
"BackupVariables/cond_2/read/Switch	RefSwitchCCN_1Conv_x0/convB10/kernelBackupVariables/cond_2/pred_id"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*4
_output_shapes"
 :??:??*
T0
?
BackupVariables/cond_2/Switch_1Switch6CCN_1Conv_x0/convB10/kernel/Initializer/random_uniformBackupVariables/cond_2/pred_id*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*4
_output_shapes"
 :??:??
?
BackupVariables/cond_2/MergeMergeBackupVariables/cond_2/Switch_1BackupVariables/cond_2/read"/device:CPU:0*
N*&
_output_shapes
:??: *
T0
?
+BackupVariables/CCN_1Conv_x0/convB10/kernel
VariableV2"/device:CPU:0*
shape:??*
dtype0*$
_output_shapes
:??
?
ABackupVariables/CCN_1Conv_x0/convB10/kernel/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB10/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
dtype0*
_output_shapes
: 
?
7BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/SwitchSwitchABackupVariables/CCN_1Conv_x0/convB10/kernel/IsVariableInitializedABackupVariables/CCN_1Conv_x0/convB10/kernel/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: : 
?
9BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/switch_tIdentity9BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
9BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/switch_fIdentity7BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
8BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/pred_idIdentityABackupVariables/CCN_1Conv_x0/convB10/kernel/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
5BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/readIdentity>BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/read/Switch:1"/device:CPU:0*
T0*$
_output_shapes
:??
?
<BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB10/kernel8BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/pred_id"/device:GPU:1*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0
?
9BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/Switch_1Switch6CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform8BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/pred_id*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0*4
_output_shapes"
 :??:??
?
6BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/MergeMerge9BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/Switch_15BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/read"/device:CPU:0*&
_output_shapes
:??: *
N*
T0
?
NBackupVariables/cond_2/read/Switch_BackupVariables/CCN_1Conv_x0/convB10/kernelSwitch6BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/MergeBackupVariables/cond_2/pred_id"/device:CPU:0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0*4
_output_shapes"
 :??:??
?
GBackupVariables/cond_2/read_BackupVariables/CCN_1Conv_x0/convB10/kernelIdentityPBackupVariables/cond_2/read/Switch_BackupVariables/CCN_1Conv_x0/convB10/kernel:1"/device:CPU:0*
T0*$
_output_shapes
:??
?
HBackupVariables/cond_2/Merge_BackupVariables/CCN_1Conv_x0/convB10/kernelMergeBackupVariables/cond_2/Switch_1GBackupVariables/cond_2/read_BackupVariables/CCN_1Conv_x0/convB10/kernel"/device:CPU:0*
N*
T0*&
_output_shapes
:??: 
?
2BackupVariables/CCN_1Conv_x0/convB10/kernel/AssignAssign+BackupVariables/CCN_1Conv_x0/convB10/kernelHBackupVariables/cond_2/Merge_BackupVariables/CCN_1Conv_x0/convB10/kernel"/device:CPU:0*
T0*$
_output_shapes
:??*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB10/kernel
?
0BackupVariables/CCN_1Conv_x0/convB10/kernel/readIdentity+BackupVariables/CCN_1Conv_x0/convB10/kernel"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??*
T0
?
'BackupVariables/IsVariableInitialized_3IsVariableInitializedCCN_1Conv_x0/convB10/bias"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
dtype0*
_output_shapes
: 
?
BackupVariables/cond_3/SwitchSwitch'BackupVariables/IsVariableInitialized_3'BackupVariables/IsVariableInitialized_3"/device:CPU:0*
_output_shapes
: : *
T0

|
BackupVariables/cond_3/switch_tIdentityBackupVariables/cond_3/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

z
BackupVariables/cond_3/switch_fIdentityBackupVariables/cond_3/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_3/pred_idIdentity'BackupVariables/IsVariableInitialized_3"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_3/readIdentity$BackupVariables/cond_3/read/Switch:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
"BackupVariables/cond_3/read/Switch	RefSwitchCCN_1Conv_x0/convB10/biasBackupVariables/cond_3/pred_id"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*"
_output_shapes
:?:?*
T0
?
BackupVariables/cond_3/Switch_1Switch+CCN_1Conv_x0/convB10/bias/Initializer/zerosBackupVariables/cond_3/pred_id*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*"
_output_shapes
:?:?
?
BackupVariables/cond_3/MergeMergeBackupVariables/cond_3/Switch_1BackupVariables/cond_3/read"/device:CPU:0*
T0*
_output_shapes
	:?: *
N
?
)BackupVariables/CCN_1Conv_x0/convB10/bias
VariableV2"/device:CPU:0*
_output_shapes	
:?*
dtype0*
shape:?
?
?BackupVariables/CCN_1Conv_x0/convB10/bias/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB10/bias"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
dtype0*
_output_shapes
: 
?
5BackupVariables/CCN_1Conv_x0/convB10/bias/cond/SwitchSwitch?BackupVariables/CCN_1Conv_x0/convB10/bias/IsVariableInitialized?BackupVariables/CCN_1Conv_x0/convB10/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
7BackupVariables/CCN_1Conv_x0/convB10/bias/cond/switch_tIdentity7BackupVariables/CCN_1Conv_x0/convB10/bias/cond/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
?
7BackupVariables/CCN_1Conv_x0/convB10/bias/cond/switch_fIdentity5BackupVariables/CCN_1Conv_x0/convB10/bias/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
6BackupVariables/CCN_1Conv_x0/convB10/bias/cond/pred_idIdentity?BackupVariables/CCN_1Conv_x0/convB10/bias/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
3BackupVariables/CCN_1Conv_x0/convB10/bias/cond/readIdentity<BackupVariables/CCN_1Conv_x0/convB10/bias/cond/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
:BackupVariables/CCN_1Conv_x0/convB10/bias/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB10/bias6BackupVariables/CCN_1Conv_x0/convB10/bias/cond/pred_id"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*"
_output_shapes
:?:?*
T0
?
7BackupVariables/CCN_1Conv_x0/convB10/bias/cond/Switch_1Switch+CCN_1Conv_x0/convB10/bias/Initializer/zeros6BackupVariables/CCN_1Conv_x0/convB10/bias/cond/pred_id*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0*"
_output_shapes
:?:?
?
4BackupVariables/CCN_1Conv_x0/convB10/bias/cond/MergeMerge7BackupVariables/CCN_1Conv_x0/convB10/bias/cond/Switch_13BackupVariables/CCN_1Conv_x0/convB10/bias/cond/read"/device:CPU:0*
N*
T0*
_output_shapes
	:?: 
?
LBackupVariables/cond_3/read/Switch_BackupVariables/CCN_1Conv_x0/convB10/biasSwitch4BackupVariables/CCN_1Conv_x0/convB10/bias/cond/MergeBackupVariables/cond_3/pred_id"/device:CPU:0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0*"
_output_shapes
:?:?
?
EBackupVariables/cond_3/read_BackupVariables/CCN_1Conv_x0/convB10/biasIdentityNBackupVariables/cond_3/read/Switch_BackupVariables/CCN_1Conv_x0/convB10/bias:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
FBackupVariables/cond_3/Merge_BackupVariables/CCN_1Conv_x0/convB10/biasMergeBackupVariables/cond_3/Switch_1EBackupVariables/cond_3/read_BackupVariables/CCN_1Conv_x0/convB10/bias"/device:CPU:0*
N*
T0*
_output_shapes
	:?: 
?
0BackupVariables/CCN_1Conv_x0/convB10/bias/AssignAssign)BackupVariables/CCN_1Conv_x0/convB10/biasFBackupVariables/cond_3/Merge_BackupVariables/CCN_1Conv_x0/convB10/bias"/device:CPU:0*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB10/bias*
_output_shapes	
:?
?
.BackupVariables/CCN_1Conv_x0/convB10/bias/readIdentity)BackupVariables/CCN_1Conv_x0/convB10/bias"/device:CPU:0*
T0*
_output_shapes	
:?*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB10/bias
?
'BackupVariables/IsVariableInitialized_4IsVariableInitializedCCN_1Conv_x0/convB20/kernel"/device:GPU:1*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
_output_shapes
: 
?
BackupVariables/cond_4/SwitchSwitch'BackupVariables/IsVariableInitialized_4'BackupVariables/IsVariableInitialized_4"/device:CPU:0*
_output_shapes
: : *
T0

|
BackupVariables/cond_4/switch_tIdentityBackupVariables/cond_4/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

z
BackupVariables/cond_4/switch_fIdentityBackupVariables/cond_4/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_4/pred_idIdentity'BackupVariables/IsVariableInitialized_4"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_4/readIdentity$BackupVariables/cond_4/read/Switch:1"/device:CPU:0*$
_output_shapes
:??*
T0
?
"BackupVariables/cond_4/read/Switch	RefSwitchCCN_1Conv_x0/convB20/kernelBackupVariables/cond_4/pred_id"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*4
_output_shapes"
 :??:??
?
BackupVariables/cond_4/Switch_1Switch6CCN_1Conv_x0/convB20/kernel/Initializer/random_uniformBackupVariables/cond_4/pred_id*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*4
_output_shapes"
 :??:??*
T0
?
BackupVariables/cond_4/MergeMergeBackupVariables/cond_4/Switch_1BackupVariables/cond_4/read"/device:CPU:0*
T0*&
_output_shapes
:??: *
N
?
+BackupVariables/CCN_1Conv_x0/convB20/kernel
VariableV2"/device:CPU:0*$
_output_shapes
:??*
shape:??*
dtype0
?
ABackupVariables/CCN_1Conv_x0/convB20/kernel/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB20/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
_output_shapes
: *
dtype0
?
7BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/SwitchSwitchABackupVariables/CCN_1Conv_x0/convB20/kernel/IsVariableInitializedABackupVariables/CCN_1Conv_x0/convB20/kernel/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: : 
?
9BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/switch_tIdentity9BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
?
9BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/switch_fIdentity7BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
8BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/pred_idIdentityABackupVariables/CCN_1Conv_x0/convB20/kernel/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
5BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/readIdentity>BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/read/Switch:1"/device:CPU:0*
T0*$
_output_shapes
:??
?
<BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB20/kernel8BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/pred_id"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*4
_output_shapes"
 :??:??*
T0
?
9BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/Switch_1Switch6CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform8BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/pred_id*
T0*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
6BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/MergeMerge9BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/Switch_15BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/read"/device:CPU:0*&
_output_shapes
:??: *
T0*
N
?
NBackupVariables/cond_4/read/Switch_BackupVariables/CCN_1Conv_x0/convB20/kernelSwitch6BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/MergeBackupVariables/cond_4/pred_id"/device:CPU:0*4
_output_shapes"
 :??:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel
?
GBackupVariables/cond_4/read_BackupVariables/CCN_1Conv_x0/convB20/kernelIdentityPBackupVariables/cond_4/read/Switch_BackupVariables/CCN_1Conv_x0/convB20/kernel:1"/device:CPU:0*$
_output_shapes
:??*
T0
?
HBackupVariables/cond_4/Merge_BackupVariables/CCN_1Conv_x0/convB20/kernelMergeBackupVariables/cond_4/Switch_1GBackupVariables/cond_4/read_BackupVariables/CCN_1Conv_x0/convB20/kernel"/device:CPU:0*
T0*
N*&
_output_shapes
:??: 
?
2BackupVariables/CCN_1Conv_x0/convB20/kernel/AssignAssign+BackupVariables/CCN_1Conv_x0/convB20/kernelHBackupVariables/cond_4/Merge_BackupVariables/CCN_1Conv_x0/convB20/kernel"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB20/kernel*
T0*$
_output_shapes
:??
?
0BackupVariables/CCN_1Conv_x0/convB20/kernel/readIdentity+BackupVariables/CCN_1Conv_x0/convB20/kernel"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB20/kernel*
T0*$
_output_shapes
:??
?
'BackupVariables/IsVariableInitialized_5IsVariableInitializedCCN_1Conv_x0/convB20/bias"/device:GPU:1*
_output_shapes
: *
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
BackupVariables/cond_5/SwitchSwitch'BackupVariables/IsVariableInitialized_5'BackupVariables/IsVariableInitialized_5"/device:CPU:0*
T0
*
_output_shapes
: : 
|
BackupVariables/cond_5/switch_tIdentityBackupVariables/cond_5/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
z
BackupVariables/cond_5/switch_fIdentityBackupVariables/cond_5/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_5/pred_idIdentity'BackupVariables/IsVariableInitialized_5"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_5/readIdentity$BackupVariables/cond_5/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
"BackupVariables/cond_5/read/Switch	RefSwitchCCN_1Conv_x0/convB20/biasBackupVariables/cond_5/pred_id"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*"
_output_shapes
:?:?
?
BackupVariables/cond_5/Switch_1Switch+CCN_1Conv_x0/convB20/bias/Initializer/zerosBackupVariables/cond_5/pred_id*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*"
_output_shapes
:?:?*
T0
?
BackupVariables/cond_5/MergeMergeBackupVariables/cond_5/Switch_1BackupVariables/cond_5/read"/device:CPU:0*
T0*
N*
_output_shapes
	:?: 
?
)BackupVariables/CCN_1Conv_x0/convB20/bias
VariableV2"/device:CPU:0*
shape:?*
_output_shapes	
:?*
dtype0
?
?BackupVariables/CCN_1Conv_x0/convB20/bias/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB20/bias"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
_output_shapes
: *
dtype0
?
5BackupVariables/CCN_1Conv_x0/convB20/bias/cond/SwitchSwitch?BackupVariables/CCN_1Conv_x0/convB20/bias/IsVariableInitialized?BackupVariables/CCN_1Conv_x0/convB20/bias/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: : 
?
7BackupVariables/CCN_1Conv_x0/convB20/bias/cond/switch_tIdentity7BackupVariables/CCN_1Conv_x0/convB20/bias/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
7BackupVariables/CCN_1Conv_x0/convB20/bias/cond/switch_fIdentity5BackupVariables/CCN_1Conv_x0/convB20/bias/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
6BackupVariables/CCN_1Conv_x0/convB20/bias/cond/pred_idIdentity?BackupVariables/CCN_1Conv_x0/convB20/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
3BackupVariables/CCN_1Conv_x0/convB20/bias/cond/readIdentity<BackupVariables/CCN_1Conv_x0/convB20/bias/cond/read/Switch:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
:BackupVariables/CCN_1Conv_x0/convB20/bias/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB20/bias6BackupVariables/CCN_1Conv_x0/convB20/bias/cond/pred_id"/device:GPU:1*"
_output_shapes
:?:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
7BackupVariables/CCN_1Conv_x0/convB20/bias/cond/Switch_1Switch+CCN_1Conv_x0/convB20/bias/Initializer/zeros6BackupVariables/CCN_1Conv_x0/convB20/bias/cond/pred_id*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*"
_output_shapes
:?:?*
T0
?
4BackupVariables/CCN_1Conv_x0/convB20/bias/cond/MergeMerge7BackupVariables/CCN_1Conv_x0/convB20/bias/cond/Switch_13BackupVariables/CCN_1Conv_x0/convB20/bias/cond/read"/device:CPU:0*
T0*
_output_shapes
	:?: *
N
?
LBackupVariables/cond_5/read/Switch_BackupVariables/CCN_1Conv_x0/convB20/biasSwitch4BackupVariables/CCN_1Conv_x0/convB20/bias/cond/MergeBackupVariables/cond_5/pred_id"/device:CPU:0*
T0*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
EBackupVariables/cond_5/read_BackupVariables/CCN_1Conv_x0/convB20/biasIdentityNBackupVariables/cond_5/read/Switch_BackupVariables/CCN_1Conv_x0/convB20/bias:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
FBackupVariables/cond_5/Merge_BackupVariables/CCN_1Conv_x0/convB20/biasMergeBackupVariables/cond_5/Switch_1EBackupVariables/cond_5/read_BackupVariables/CCN_1Conv_x0/convB20/bias"/device:CPU:0*
T0*
_output_shapes
	:?: *
N
?
0BackupVariables/CCN_1Conv_x0/convB20/bias/AssignAssign)BackupVariables/CCN_1Conv_x0/convB20/biasFBackupVariables/cond_5/Merge_BackupVariables/CCN_1Conv_x0/convB20/bias"/device:CPU:0*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB20/bias*
_output_shapes	
:?
?
.BackupVariables/CCN_1Conv_x0/convB20/bias/readIdentity)BackupVariables/CCN_1Conv_x0/convB20/bias"/device:CPU:0*
_output_shapes	
:?*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB20/bias*
T0
?
'BackupVariables/IsVariableInitialized_6IsVariableInitializedCCN_1Conv_x0/convA11/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
dtype0*
_output_shapes
: 
?
BackupVariables/cond_6/SwitchSwitch'BackupVariables/IsVariableInitialized_6'BackupVariables/IsVariableInitialized_6"/device:CPU:0*
T0
*
_output_shapes
: : 
|
BackupVariables/cond_6/switch_tIdentityBackupVariables/cond_6/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

z
BackupVariables/cond_6/switch_fIdentityBackupVariables/cond_6/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_6/pred_idIdentity'BackupVariables/IsVariableInitialized_6"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_6/readIdentity$BackupVariables/cond_6/read/Switch:1"/device:CPU:0*$
_output_shapes
:??*
T0
?
"BackupVariables/cond_6/read/Switch	RefSwitchCCN_1Conv_x0/convA11/kernelBackupVariables/cond_6/pred_id"/device:GPU:1*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0
?
BackupVariables/cond_6/Switch_1Switch6CCN_1Conv_x0/convA11/kernel/Initializer/random_uniformBackupVariables/cond_6/pred_id*
T0*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
BackupVariables/cond_6/MergeMergeBackupVariables/cond_6/Switch_1BackupVariables/cond_6/read"/device:CPU:0*
N*&
_output_shapes
:??: *
T0
?
+BackupVariables/CCN_1Conv_x0/convA11/kernel
VariableV2"/device:CPU:0*$
_output_shapes
:??*
dtype0*
shape:??
?
ABackupVariables/CCN_1Conv_x0/convA11/kernel/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convA11/kernel"/device:GPU:1*
_output_shapes
: *
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
7BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/SwitchSwitchABackupVariables/CCN_1Conv_x0/convA11/kernel/IsVariableInitializedABackupVariables/CCN_1Conv_x0/convA11/kernel/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
9BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/switch_tIdentity9BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
9BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/switch_fIdentity7BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
8BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/pred_idIdentityABackupVariables/CCN_1Conv_x0/convA11/kernel/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
5BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/readIdentity>BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/read/Switch:1"/device:CPU:0*$
_output_shapes
:??*
T0
?
<BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/read/Switch	RefSwitchCCN_1Conv_x0/convA11/kernel8BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/pred_id"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0*4
_output_shapes"
 :??:??
?
9BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/Switch_1Switch6CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform8BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/pred_id*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0*4
_output_shapes"
 :??:??
?
6BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/MergeMerge9BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/Switch_15BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/read"/device:CPU:0*&
_output_shapes
:??: *
N*
T0
?
NBackupVariables/cond_6/read/Switch_BackupVariables/CCN_1Conv_x0/convA11/kernelSwitch6BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/MergeBackupVariables/cond_6/pred_id"/device:CPU:0*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*4
_output_shapes"
 :??:??
?
GBackupVariables/cond_6/read_BackupVariables/CCN_1Conv_x0/convA11/kernelIdentityPBackupVariables/cond_6/read/Switch_BackupVariables/CCN_1Conv_x0/convA11/kernel:1"/device:CPU:0*
T0*$
_output_shapes
:??
?
HBackupVariables/cond_6/Merge_BackupVariables/CCN_1Conv_x0/convA11/kernelMergeBackupVariables/cond_6/Switch_1GBackupVariables/cond_6/read_BackupVariables/CCN_1Conv_x0/convA11/kernel"/device:CPU:0*
N*&
_output_shapes
:??: *
T0
?
2BackupVariables/CCN_1Conv_x0/convA11/kernel/AssignAssign+BackupVariables/CCN_1Conv_x0/convA11/kernelHBackupVariables/cond_6/Merge_BackupVariables/CCN_1Conv_x0/convA11/kernel"/device:CPU:0*$
_output_shapes
:??*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convA11/kernel*
T0
?
0BackupVariables/CCN_1Conv_x0/convA11/kernel/readIdentity+BackupVariables/CCN_1Conv_x0/convA11/kernel"/device:CPU:0*
T0*$
_output_shapes
:??*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convA11/kernel
?
'BackupVariables/IsVariableInitialized_7IsVariableInitializedCCN_1Conv_x0/convA11/bias"/device:GPU:1*
_output_shapes
: *
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias
?
BackupVariables/cond_7/SwitchSwitch'BackupVariables/IsVariableInitialized_7'BackupVariables/IsVariableInitialized_7"/device:CPU:0*
_output_shapes
: : *
T0

|
BackupVariables/cond_7/switch_tIdentityBackupVariables/cond_7/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
z
BackupVariables/cond_7/switch_fIdentityBackupVariables/cond_7/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_7/pred_idIdentity'BackupVariables/IsVariableInitialized_7"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_7/readIdentity$BackupVariables/cond_7/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
"BackupVariables/cond_7/read/Switch	RefSwitchCCN_1Conv_x0/convA11/biasBackupVariables/cond_7/pred_id"/device:GPU:1*
T0*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias
?
BackupVariables/cond_7/Switch_1Switch+CCN_1Conv_x0/convA11/bias/Initializer/zerosBackupVariables/cond_7/pred_id*"
_output_shapes
:?:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias
?
BackupVariables/cond_7/MergeMergeBackupVariables/cond_7/Switch_1BackupVariables/cond_7/read"/device:CPU:0*
T0*
N*
_output_shapes
	:?: 
?
)BackupVariables/CCN_1Conv_x0/convA11/bias
VariableV2"/device:CPU:0*
_output_shapes	
:?*
shape:?*
dtype0
?
?BackupVariables/CCN_1Conv_x0/convA11/bias/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convA11/bias"/device:GPU:1*
dtype0*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convA11/bias
?
5BackupVariables/CCN_1Conv_x0/convA11/bias/cond/SwitchSwitch?BackupVariables/CCN_1Conv_x0/convA11/bias/IsVariableInitialized?BackupVariables/CCN_1Conv_x0/convA11/bias/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: : 
?
7BackupVariables/CCN_1Conv_x0/convA11/bias/cond/switch_tIdentity7BackupVariables/CCN_1Conv_x0/convA11/bias/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
7BackupVariables/CCN_1Conv_x0/convA11/bias/cond/switch_fIdentity5BackupVariables/CCN_1Conv_x0/convA11/bias/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
6BackupVariables/CCN_1Conv_x0/convA11/bias/cond/pred_idIdentity?BackupVariables/CCN_1Conv_x0/convA11/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
3BackupVariables/CCN_1Conv_x0/convA11/bias/cond/readIdentity<BackupVariables/CCN_1Conv_x0/convA11/bias/cond/read/Switch:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
:BackupVariables/CCN_1Conv_x0/convA11/bias/cond/read/Switch	RefSwitchCCN_1Conv_x0/convA11/bias6BackupVariables/CCN_1Conv_x0/convA11/bias/cond/pred_id"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0*"
_output_shapes
:?:?
?
7BackupVariables/CCN_1Conv_x0/convA11/bias/cond/Switch_1Switch+CCN_1Conv_x0/convA11/bias/Initializer/zeros6BackupVariables/CCN_1Conv_x0/convA11/bias/cond/pred_id*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0*"
_output_shapes
:?:?
?
4BackupVariables/CCN_1Conv_x0/convA11/bias/cond/MergeMerge7BackupVariables/CCN_1Conv_x0/convA11/bias/cond/Switch_13BackupVariables/CCN_1Conv_x0/convA11/bias/cond/read"/device:CPU:0*
T0*
N*
_output_shapes
	:?: 
?
LBackupVariables/cond_7/read/Switch_BackupVariables/CCN_1Conv_x0/convA11/biasSwitch4BackupVariables/CCN_1Conv_x0/convA11/bias/cond/MergeBackupVariables/cond_7/pred_id"/device:CPU:0*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*"
_output_shapes
:?:?
?
EBackupVariables/cond_7/read_BackupVariables/CCN_1Conv_x0/convA11/biasIdentityNBackupVariables/cond_7/read/Switch_BackupVariables/CCN_1Conv_x0/convA11/bias:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
FBackupVariables/cond_7/Merge_BackupVariables/CCN_1Conv_x0/convA11/biasMergeBackupVariables/cond_7/Switch_1EBackupVariables/cond_7/read_BackupVariables/CCN_1Conv_x0/convA11/bias"/device:CPU:0*
N*
T0*
_output_shapes
	:?: 
?
0BackupVariables/CCN_1Conv_x0/convA11/bias/AssignAssign)BackupVariables/CCN_1Conv_x0/convA11/biasFBackupVariables/cond_7/Merge_BackupVariables/CCN_1Conv_x0/convA11/bias"/device:CPU:0*
T0*
_output_shapes	
:?*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convA11/bias
?
.BackupVariables/CCN_1Conv_x0/convA11/bias/readIdentity)BackupVariables/CCN_1Conv_x0/convA11/bias"/device:CPU:0*
T0*
_output_shapes	
:?*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convA11/bias
?
'BackupVariables/IsVariableInitialized_8IsVariableInitializedCCN_1Conv_x0/convB11/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
dtype0*
_output_shapes
: 
?
BackupVariables/cond_8/SwitchSwitch'BackupVariables/IsVariableInitialized_8'BackupVariables/IsVariableInitialized_8"/device:CPU:0*
_output_shapes
: : *
T0

|
BackupVariables/cond_8/switch_tIdentityBackupVariables/cond_8/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
z
BackupVariables/cond_8/switch_fIdentityBackupVariables/cond_8/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_8/pred_idIdentity'BackupVariables/IsVariableInitialized_8"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_8/readIdentity$BackupVariables/cond_8/read/Switch:1"/device:CPU:0*$
_output_shapes
:??*
T0
?
"BackupVariables/cond_8/read/Switch	RefSwitchCCN_1Conv_x0/convB11/kernelBackupVariables/cond_8/pred_id"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*4
_output_shapes"
 :??:??
?
BackupVariables/cond_8/Switch_1Switch6CCN_1Conv_x0/convB11/kernel/Initializer/random_uniformBackupVariables/cond_8/pred_id*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0*4
_output_shapes"
 :??:??
?
BackupVariables/cond_8/MergeMergeBackupVariables/cond_8/Switch_1BackupVariables/cond_8/read"/device:CPU:0*
T0*&
_output_shapes
:??: *
N
?
+BackupVariables/CCN_1Conv_x0/convB11/kernel
VariableV2"/device:CPU:0*$
_output_shapes
:??*
shape:??*
dtype0
?
ABackupVariables/CCN_1Conv_x0/convB11/kernel/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB11/kernel"/device:GPU:1*
dtype0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
_output_shapes
: 
?
7BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/SwitchSwitchABackupVariables/CCN_1Conv_x0/convB11/kernel/IsVariableInitializedABackupVariables/CCN_1Conv_x0/convB11/kernel/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
9BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/switch_tIdentity9BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
9BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/switch_fIdentity7BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
8BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/pred_idIdentityABackupVariables/CCN_1Conv_x0/convB11/kernel/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
5BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/readIdentity>BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/read/Switch:1"/device:CPU:0*
T0*$
_output_shapes
:??
?
<BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB11/kernel8BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/pred_id"/device:GPU:1*
T0*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
9BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/Switch_1Switch6CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform8BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/pred_id*
T0*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
6BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/MergeMerge9BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/Switch_15BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/read"/device:CPU:0*&
_output_shapes
:??: *
N*
T0
?
NBackupVariables/cond_8/read/Switch_BackupVariables/CCN_1Conv_x0/convB11/kernelSwitch6BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/MergeBackupVariables/cond_8/pred_id"/device:CPU:0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0*4
_output_shapes"
 :??:??
?
GBackupVariables/cond_8/read_BackupVariables/CCN_1Conv_x0/convB11/kernelIdentityPBackupVariables/cond_8/read/Switch_BackupVariables/CCN_1Conv_x0/convB11/kernel:1"/device:CPU:0*$
_output_shapes
:??*
T0
?
HBackupVariables/cond_8/Merge_BackupVariables/CCN_1Conv_x0/convB11/kernelMergeBackupVariables/cond_8/Switch_1GBackupVariables/cond_8/read_BackupVariables/CCN_1Conv_x0/convB11/kernel"/device:CPU:0*
T0*
N*&
_output_shapes
:??: 
?
2BackupVariables/CCN_1Conv_x0/convB11/kernel/AssignAssign+BackupVariables/CCN_1Conv_x0/convB11/kernelHBackupVariables/cond_8/Merge_BackupVariables/CCN_1Conv_x0/convB11/kernel"/device:CPU:0*$
_output_shapes
:??*
T0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB11/kernel
?
0BackupVariables/CCN_1Conv_x0/convB11/kernel/readIdentity+BackupVariables/CCN_1Conv_x0/convB11/kernel"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB11/kernel*$
_output_shapes
:??*
T0
?
'BackupVariables/IsVariableInitialized_9IsVariableInitializedCCN_1Conv_x0/convB11/bias"/device:GPU:1*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes
: 
?
BackupVariables/cond_9/SwitchSwitch'BackupVariables/IsVariableInitialized_9'BackupVariables/IsVariableInitialized_9"/device:CPU:0*
_output_shapes
: : *
T0

|
BackupVariables/cond_9/switch_tIdentityBackupVariables/cond_9/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

z
BackupVariables/cond_9/switch_fIdentityBackupVariables/cond_9/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_9/pred_idIdentity'BackupVariables/IsVariableInitialized_9"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_9/readIdentity$BackupVariables/cond_9/read/Switch:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
"BackupVariables/cond_9/read/Switch	RefSwitchCCN_1Conv_x0/convB11/biasBackupVariables/cond_9/pred_id"/device:GPU:1*
T0*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
BackupVariables/cond_9/Switch_1Switch+CCN_1Conv_x0/convB11/bias/Initializer/zerosBackupVariables/cond_9/pred_id*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*"
_output_shapes
:?:?
?
BackupVariables/cond_9/MergeMergeBackupVariables/cond_9/Switch_1BackupVariables/cond_9/read"/device:CPU:0*
_output_shapes
	:?: *
N*
T0
?
)BackupVariables/CCN_1Conv_x0/convB11/bias
VariableV2"/device:CPU:0*
_output_shapes	
:?*
dtype0*
shape:?
?
?BackupVariables/CCN_1Conv_x0/convB11/bias/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB11/bias"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes
: *
dtype0
?
5BackupVariables/CCN_1Conv_x0/convB11/bias/cond/SwitchSwitch?BackupVariables/CCN_1Conv_x0/convB11/bias/IsVariableInitialized?BackupVariables/CCN_1Conv_x0/convB11/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
7BackupVariables/CCN_1Conv_x0/convB11/bias/cond/switch_tIdentity7BackupVariables/CCN_1Conv_x0/convB11/bias/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
7BackupVariables/CCN_1Conv_x0/convB11/bias/cond/switch_fIdentity5BackupVariables/CCN_1Conv_x0/convB11/bias/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
6BackupVariables/CCN_1Conv_x0/convB11/bias/cond/pred_idIdentity?BackupVariables/CCN_1Conv_x0/convB11/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
3BackupVariables/CCN_1Conv_x0/convB11/bias/cond/readIdentity<BackupVariables/CCN_1Conv_x0/convB11/bias/cond/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
:BackupVariables/CCN_1Conv_x0/convB11/bias/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB11/bias6BackupVariables/CCN_1Conv_x0/convB11/bias/cond/pred_id"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*"
_output_shapes
:?:?
?
7BackupVariables/CCN_1Conv_x0/convB11/bias/cond/Switch_1Switch+CCN_1Conv_x0/convB11/bias/Initializer/zeros6BackupVariables/CCN_1Conv_x0/convB11/bias/cond/pred_id*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
T0
?
4BackupVariables/CCN_1Conv_x0/convB11/bias/cond/MergeMerge7BackupVariables/CCN_1Conv_x0/convB11/bias/cond/Switch_13BackupVariables/CCN_1Conv_x0/convB11/bias/cond/read"/device:CPU:0*
N*
_output_shapes
	:?: *
T0
?
LBackupVariables/cond_9/read/Switch_BackupVariables/CCN_1Conv_x0/convB11/biasSwitch4BackupVariables/CCN_1Conv_x0/convB11/bias/cond/MergeBackupVariables/cond_9/pred_id"/device:CPU:0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*"
_output_shapes
:?:?*
T0
?
EBackupVariables/cond_9/read_BackupVariables/CCN_1Conv_x0/convB11/biasIdentityNBackupVariables/cond_9/read/Switch_BackupVariables/CCN_1Conv_x0/convB11/bias:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
FBackupVariables/cond_9/Merge_BackupVariables/CCN_1Conv_x0/convB11/biasMergeBackupVariables/cond_9/Switch_1EBackupVariables/cond_9/read_BackupVariables/CCN_1Conv_x0/convB11/bias"/device:CPU:0*
N*
_output_shapes
	:?: *
T0
?
0BackupVariables/CCN_1Conv_x0/convB11/bias/AssignAssign)BackupVariables/CCN_1Conv_x0/convB11/biasFBackupVariables/cond_9/Merge_BackupVariables/CCN_1Conv_x0/convB11/bias"/device:CPU:0*
_output_shapes	
:?*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB11/bias
?
.BackupVariables/CCN_1Conv_x0/convB11/bias/readIdentity)BackupVariables/CCN_1Conv_x0/convB11/bias"/device:CPU:0*
_output_shapes	
:?*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB11/bias*
T0
?
(BackupVariables/IsVariableInitialized_10IsVariableInitializedCCN_1Conv_x0/convB21/kernel"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
dtype0*
_output_shapes
: 
?
BackupVariables/cond_10/SwitchSwitch(BackupVariables/IsVariableInitialized_10(BackupVariables/IsVariableInitialized_10"/device:CPU:0*
T0
*
_output_shapes
: : 
~
 BackupVariables/cond_10/switch_tIdentity BackupVariables/cond_10/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

|
 BackupVariables/cond_10/switch_fIdentityBackupVariables/cond_10/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_10/pred_idIdentity(BackupVariables/IsVariableInitialized_10"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_10/readIdentity%BackupVariables/cond_10/read/Switch:1"/device:CPU:0*
T0*$
_output_shapes
:??
?
#BackupVariables/cond_10/read/Switch	RefSwitchCCN_1Conv_x0/convB21/kernelBackupVariables/cond_10/pred_id"/device:GPU:1*
T0*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
 BackupVariables/cond_10/Switch_1Switch6CCN_1Conv_x0/convB21/kernel/Initializer/random_uniformBackupVariables/cond_10/pred_id*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
T0
?
BackupVariables/cond_10/MergeMerge BackupVariables/cond_10/Switch_1BackupVariables/cond_10/read"/device:CPU:0*
N*&
_output_shapes
:??: *
T0
?
+BackupVariables/CCN_1Conv_x0/convB21/kernel
VariableV2"/device:CPU:0*$
_output_shapes
:??*
shape:??*
dtype0
?
ABackupVariables/CCN_1Conv_x0/convB21/kernel/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB21/kernel"/device:GPU:1*
_output_shapes
: *.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
dtype0
?
7BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/SwitchSwitchABackupVariables/CCN_1Conv_x0/convB21/kernel/IsVariableInitializedABackupVariables/CCN_1Conv_x0/convB21/kernel/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
9BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/switch_tIdentity9BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
?
9BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/switch_fIdentity7BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
8BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/pred_idIdentityABackupVariables/CCN_1Conv_x0/convB21/kernel/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
5BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/readIdentity>BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/read/Switch:1"/device:CPU:0*
T0*$
_output_shapes
:??
?
<BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB21/kernel8BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/pred_id"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*4
_output_shapes"
 :??:??
?
9BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/Switch_1Switch6CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform8BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/pred_id*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
T0*4
_output_shapes"
 :??:??
?
6BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/MergeMerge9BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/Switch_15BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/read"/device:CPU:0*
N*
T0*&
_output_shapes
:??: 
?
OBackupVariables/cond_10/read/Switch_BackupVariables/CCN_1Conv_x0/convB21/kernelSwitch6BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/MergeBackupVariables/cond_10/pred_id"/device:CPU:0*
T0*4
_output_shapes"
 :??:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
HBackupVariables/cond_10/read_BackupVariables/CCN_1Conv_x0/convB21/kernelIdentityQBackupVariables/cond_10/read/Switch_BackupVariables/CCN_1Conv_x0/convB21/kernel:1"/device:CPU:0*$
_output_shapes
:??*
T0
?
IBackupVariables/cond_10/Merge_BackupVariables/CCN_1Conv_x0/convB21/kernelMerge BackupVariables/cond_10/Switch_1HBackupVariables/cond_10/read_BackupVariables/CCN_1Conv_x0/convB21/kernel"/device:CPU:0*
T0*
N*&
_output_shapes
:??: 
?
2BackupVariables/CCN_1Conv_x0/convB21/kernel/AssignAssign+BackupVariables/CCN_1Conv_x0/convB21/kernelIBackupVariables/cond_10/Merge_BackupVariables/CCN_1Conv_x0/convB21/kernel"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB21/kernel*
T0*$
_output_shapes
:??
?
0BackupVariables/CCN_1Conv_x0/convB21/kernel/readIdentity+BackupVariables/CCN_1Conv_x0/convB21/kernel"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??*
T0
?
(BackupVariables/IsVariableInitialized_11IsVariableInitializedCCN_1Conv_x0/convB21/bias"/device:GPU:1*
dtype0*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes
: 
?
BackupVariables/cond_11/SwitchSwitch(BackupVariables/IsVariableInitialized_11(BackupVariables/IsVariableInitialized_11"/device:CPU:0*
T0
*
_output_shapes
: : 
~
 BackupVariables/cond_11/switch_tIdentity BackupVariables/cond_11/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

|
 BackupVariables/cond_11/switch_fIdentityBackupVariables/cond_11/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_11/pred_idIdentity(BackupVariables/IsVariableInitialized_11"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_11/readIdentity%BackupVariables/cond_11/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
#BackupVariables/cond_11/read/Switch	RefSwitchCCN_1Conv_x0/convB21/biasBackupVariables/cond_11/pred_id"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0*"
_output_shapes
:?:?
?
 BackupVariables/cond_11/Switch_1Switch+CCN_1Conv_x0/convB21/bias/Initializer/zerosBackupVariables/cond_11/pred_id*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0
?
BackupVariables/cond_11/MergeMerge BackupVariables/cond_11/Switch_1BackupVariables/cond_11/read"/device:CPU:0*
_output_shapes
	:?: *
T0*
N
?
)BackupVariables/CCN_1Conv_x0/convB21/bias
VariableV2"/device:CPU:0*
dtype0*
shape:?*
_output_shapes	
:?
?
?BackupVariables/CCN_1Conv_x0/convB21/bias/IsVariableInitializedIsVariableInitializedCCN_1Conv_x0/convB21/bias"/device:GPU:1*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
dtype0
?
5BackupVariables/CCN_1Conv_x0/convB21/bias/cond/SwitchSwitch?BackupVariables/CCN_1Conv_x0/convB21/bias/IsVariableInitialized?BackupVariables/CCN_1Conv_x0/convB21/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
7BackupVariables/CCN_1Conv_x0/convB21/bias/cond/switch_tIdentity7BackupVariables/CCN_1Conv_x0/convB21/bias/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
7BackupVariables/CCN_1Conv_x0/convB21/bias/cond/switch_fIdentity5BackupVariables/CCN_1Conv_x0/convB21/bias/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
6BackupVariables/CCN_1Conv_x0/convB21/bias/cond/pred_idIdentity?BackupVariables/CCN_1Conv_x0/convB21/bias/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
3BackupVariables/CCN_1Conv_x0/convB21/bias/cond/readIdentity<BackupVariables/CCN_1Conv_x0/convB21/bias/cond/read/Switch:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
:BackupVariables/CCN_1Conv_x0/convB21/bias/cond/read/Switch	RefSwitchCCN_1Conv_x0/convB21/bias6BackupVariables/CCN_1Conv_x0/convB21/bias/cond/pred_id"/device:GPU:1*
T0*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
7BackupVariables/CCN_1Conv_x0/convB21/bias/cond/Switch_1Switch+CCN_1Conv_x0/convB21/bias/Initializer/zeros6BackupVariables/CCN_1Conv_x0/convB21/bias/cond/pred_id*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*"
_output_shapes
:?:?*
T0
?
4BackupVariables/CCN_1Conv_x0/convB21/bias/cond/MergeMerge7BackupVariables/CCN_1Conv_x0/convB21/bias/cond/Switch_13BackupVariables/CCN_1Conv_x0/convB21/bias/cond/read"/device:CPU:0*
T0*
_output_shapes
	:?: *
N
?
MBackupVariables/cond_11/read/Switch_BackupVariables/CCN_1Conv_x0/convB21/biasSwitch4BackupVariables/CCN_1Conv_x0/convB21/bias/cond/MergeBackupVariables/cond_11/pred_id"/device:CPU:0*"
_output_shapes
:?:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0
?
FBackupVariables/cond_11/read_BackupVariables/CCN_1Conv_x0/convB21/biasIdentityOBackupVariables/cond_11/read/Switch_BackupVariables/CCN_1Conv_x0/convB21/bias:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
GBackupVariables/cond_11/Merge_BackupVariables/CCN_1Conv_x0/convB21/biasMerge BackupVariables/cond_11/Switch_1FBackupVariables/cond_11/read_BackupVariables/CCN_1Conv_x0/convB21/bias"/device:CPU:0*
_output_shapes
	:?: *
N*
T0
?
0BackupVariables/CCN_1Conv_x0/convB21/bias/AssignAssign)BackupVariables/CCN_1Conv_x0/convB21/biasGBackupVariables/cond_11/Merge_BackupVariables/CCN_1Conv_x0/convB21/bias"/device:CPU:0*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB21/bias*
_output_shapes	
:?
?
.BackupVariables/CCN_1Conv_x0/convB21/bias/readIdentity)BackupVariables/CCN_1Conv_x0/convB21/bias"/device:CPU:0*
_output_shapes	
:?*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB21/bias
?
(BackupVariables/IsVariableInitialized_12IsVariableInitializedConv_out__/beta"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
_output_shapes
: *
dtype0
?
BackupVariables/cond_12/SwitchSwitch(BackupVariables/IsVariableInitialized_12(BackupVariables/IsVariableInitialized_12"/device:CPU:0*
T0
*
_output_shapes
: : 
~
 BackupVariables/cond_12/switch_tIdentity BackupVariables/cond_12/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

|
 BackupVariables/cond_12/switch_fIdentityBackupVariables/cond_12/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_12/pred_idIdentity(BackupVariables/IsVariableInitialized_12"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_12/readIdentity%BackupVariables/cond_12/read/Switch:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
#BackupVariables/cond_12/read/Switch	RefSwitchConv_out__/betaBackupVariables/cond_12/pred_id"/device:GPU:1*
T0*"
_class
loc:@Conv_out__/beta*"
_output_shapes
:?:?
?
 BackupVariables/cond_12/Switch_1Switch!Conv_out__/beta/Initializer/zerosBackupVariables/cond_12/pred_id*"
_class
loc:@Conv_out__/beta*
T0*"
_output_shapes
:?:?
?
BackupVariables/cond_12/MergeMerge BackupVariables/cond_12/Switch_1BackupVariables/cond_12/read"/device:CPU:0*
_output_shapes
	:?: *
N*
T0
x
BackupVariables/Conv_out__/beta
VariableV2"/device:CPU:0*
dtype0*
shape:?*
_output_shapes	
:?
?
5BackupVariables/Conv_out__/beta/IsVariableInitializedIsVariableInitializedConv_out__/beta"/device:GPU:1*
dtype0*
_output_shapes
: *"
_class
loc:@Conv_out__/beta
?
+BackupVariables/Conv_out__/beta/cond/SwitchSwitch5BackupVariables/Conv_out__/beta/IsVariableInitialized5BackupVariables/Conv_out__/beta/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
-BackupVariables/Conv_out__/beta/cond/switch_tIdentity-BackupVariables/Conv_out__/beta/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
-BackupVariables/Conv_out__/beta/cond/switch_fIdentity+BackupVariables/Conv_out__/beta/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
,BackupVariables/Conv_out__/beta/cond/pred_idIdentity5BackupVariables/Conv_out__/beta/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
)BackupVariables/Conv_out__/beta/cond/readIdentity2BackupVariables/Conv_out__/beta/cond/read/Switch:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
0BackupVariables/Conv_out__/beta/cond/read/Switch	RefSwitchConv_out__/beta,BackupVariables/Conv_out__/beta/cond/pred_id"/device:GPU:1*"
_class
loc:@Conv_out__/beta*"
_output_shapes
:?:?*
T0
?
-BackupVariables/Conv_out__/beta/cond/Switch_1Switch!Conv_out__/beta/Initializer/zeros,BackupVariables/Conv_out__/beta/cond/pred_id*"
_class
loc:@Conv_out__/beta*"
_output_shapes
:?:?*
T0
?
*BackupVariables/Conv_out__/beta/cond/MergeMerge-BackupVariables/Conv_out__/beta/cond/Switch_1)BackupVariables/Conv_out__/beta/cond/read"/device:CPU:0*
T0*
_output_shapes
	:?: *
N
?
CBackupVariables/cond_12/read/Switch_BackupVariables/Conv_out__/betaSwitch*BackupVariables/Conv_out__/beta/cond/MergeBackupVariables/cond_12/pred_id"/device:CPU:0*"
_class
loc:@Conv_out__/beta*
T0*"
_output_shapes
:?:?
?
<BackupVariables/cond_12/read_BackupVariables/Conv_out__/betaIdentityEBackupVariables/cond_12/read/Switch_BackupVariables/Conv_out__/beta:1"/device:CPU:0*
T0*
_output_shapes	
:?
?
=BackupVariables/cond_12/Merge_BackupVariables/Conv_out__/betaMerge BackupVariables/cond_12/Switch_1<BackupVariables/cond_12/read_BackupVariables/Conv_out__/beta"/device:CPU:0*
N*
T0*
_output_shapes
	:?: 
?
&BackupVariables/Conv_out__/beta/AssignAssignBackupVariables/Conv_out__/beta=BackupVariables/cond_12/Merge_BackupVariables/Conv_out__/beta"/device:CPU:0*
_output_shapes	
:?*
T0*2
_class(
&$loc:@BackupVariables/Conv_out__/beta
?
$BackupVariables/Conv_out__/beta/readIdentityBackupVariables/Conv_out__/beta"/device:CPU:0*2
_class(
&$loc:@BackupVariables/Conv_out__/beta*
T0*
_output_shapes	
:?
?
(BackupVariables/IsVariableInitialized_13IsVariableInitializedConv_out__/gamma"/device:GPU:1*
dtype0*#
_class
loc:@Conv_out__/gamma*
_output_shapes
: 
?
BackupVariables/cond_13/SwitchSwitch(BackupVariables/IsVariableInitialized_13(BackupVariables/IsVariableInitialized_13"/device:CPU:0*
_output_shapes
: : *
T0

~
 BackupVariables/cond_13/switch_tIdentity BackupVariables/cond_13/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

|
 BackupVariables/cond_13/switch_fIdentityBackupVariables/cond_13/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_13/pred_idIdentity(BackupVariables/IsVariableInitialized_13"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_13/readIdentity%BackupVariables/cond_13/read/Switch:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
#BackupVariables/cond_13/read/Switch	RefSwitchConv_out__/gammaBackupVariables/cond_13/pred_id"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*"
_output_shapes
:?:?*
T0
?
 BackupVariables/cond_13/Switch_1Switch!Conv_out__/gamma/Initializer/onesBackupVariables/cond_13/pred_id*#
_class
loc:@Conv_out__/gamma*"
_output_shapes
:?:?*
T0
?
BackupVariables/cond_13/MergeMerge BackupVariables/cond_13/Switch_1BackupVariables/cond_13/read"/device:CPU:0*
T0*
_output_shapes
	:?: *
N
y
 BackupVariables/Conv_out__/gamma
VariableV2"/device:CPU:0*
shape:?*
_output_shapes	
:?*
dtype0
?
6BackupVariables/Conv_out__/gamma/IsVariableInitializedIsVariableInitializedConv_out__/gamma"/device:GPU:1*
dtype0*#
_class
loc:@Conv_out__/gamma*
_output_shapes
: 
?
,BackupVariables/Conv_out__/gamma/cond/SwitchSwitch6BackupVariables/Conv_out__/gamma/IsVariableInitialized6BackupVariables/Conv_out__/gamma/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: : 
?
.BackupVariables/Conv_out__/gamma/cond/switch_tIdentity.BackupVariables/Conv_out__/gamma/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
.BackupVariables/Conv_out__/gamma/cond/switch_fIdentity,BackupVariables/Conv_out__/gamma/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
-BackupVariables/Conv_out__/gamma/cond/pred_idIdentity6BackupVariables/Conv_out__/gamma/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
*BackupVariables/Conv_out__/gamma/cond/readIdentity3BackupVariables/Conv_out__/gamma/cond/read/Switch:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
1BackupVariables/Conv_out__/gamma/cond/read/Switch	RefSwitchConv_out__/gamma-BackupVariables/Conv_out__/gamma/cond/pred_id"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
T0*"
_output_shapes
:?:?
?
.BackupVariables/Conv_out__/gamma/cond/Switch_1Switch!Conv_out__/gamma/Initializer/ones-BackupVariables/Conv_out__/gamma/cond/pred_id*"
_output_shapes
:?:?*
T0*#
_class
loc:@Conv_out__/gamma
?
+BackupVariables/Conv_out__/gamma/cond/MergeMerge.BackupVariables/Conv_out__/gamma/cond/Switch_1*BackupVariables/Conv_out__/gamma/cond/read"/device:CPU:0*
_output_shapes
	:?: *
T0*
N
?
DBackupVariables/cond_13/read/Switch_BackupVariables/Conv_out__/gammaSwitch+BackupVariables/Conv_out__/gamma/cond/MergeBackupVariables/cond_13/pred_id"/device:CPU:0*#
_class
loc:@Conv_out__/gamma*"
_output_shapes
:?:?*
T0
?
=BackupVariables/cond_13/read_BackupVariables/Conv_out__/gammaIdentityFBackupVariables/cond_13/read/Switch_BackupVariables/Conv_out__/gamma:1"/device:CPU:0*
_output_shapes	
:?*
T0
?
>BackupVariables/cond_13/Merge_BackupVariables/Conv_out__/gammaMerge BackupVariables/cond_13/Switch_1=BackupVariables/cond_13/read_BackupVariables/Conv_out__/gamma"/device:CPU:0*
T0*
_output_shapes
	:?: *
N
?
'BackupVariables/Conv_out__/gamma/AssignAssign BackupVariables/Conv_out__/gamma>BackupVariables/cond_13/Merge_BackupVariables/Conv_out__/gamma"/device:CPU:0*
_output_shapes	
:?*
T0*3
_class)
'%loc:@BackupVariables/Conv_out__/gamma
?
%BackupVariables/Conv_out__/gamma/readIdentity BackupVariables/Conv_out__/gamma"/device:CPU:0*
T0*3
_class)
'%loc:@BackupVariables/Conv_out__/gamma*
_output_shapes	
:?
?
(BackupVariables/IsVariableInitialized_14IsVariableInitialized"Reconstruction_Output/dense/kernel"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
dtype0*
_output_shapes
: 
?
BackupVariables/cond_14/SwitchSwitch(BackupVariables/IsVariableInitialized_14(BackupVariables/IsVariableInitialized_14"/device:CPU:0*
T0
*
_output_shapes
: : 
~
 BackupVariables/cond_14/switch_tIdentity BackupVariables/cond_14/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
|
 BackupVariables/cond_14/switch_fIdentityBackupVariables/cond_14/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_14/pred_idIdentity(BackupVariables/IsVariableInitialized_14"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_14/readIdentity%BackupVariables/cond_14/read/Switch:1"/device:CPU:0*
_output_shapes
:	?*
T0
?
#BackupVariables/cond_14/read/Switch	RefSwitch"Reconstruction_Output/dense/kernelBackupVariables/cond_14/pred_id"/device:GPU:1**
_output_shapes
:	?:	?*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
 BackupVariables/cond_14/Switch_1Switch=Reconstruction_Output/dense/kernel/Initializer/random_uniformBackupVariables/cond_14/pred_id*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
T0**
_output_shapes
:	?:	?
?
BackupVariables/cond_14/MergeMerge BackupVariables/cond_14/Switch_1BackupVariables/cond_14/read"/device:CPU:0*!
_output_shapes
:	?: *
N*
T0
?
2BackupVariables/Reconstruction_Output/dense/kernel
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:	?*
shape:	?
?
HBackupVariables/Reconstruction_Output/dense/kernel/IsVariableInitializedIsVariableInitialized"Reconstruction_Output/dense/kernel"/device:GPU:1*
_output_shapes
: *
dtype0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
>BackupVariables/Reconstruction_Output/dense/kernel/cond/SwitchSwitchHBackupVariables/Reconstruction_Output/dense/kernel/IsVariableInitializedHBackupVariables/Reconstruction_Output/dense/kernel/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
@BackupVariables/Reconstruction_Output/dense/kernel/cond/switch_tIdentity@BackupVariables/Reconstruction_Output/dense/kernel/cond/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
?
@BackupVariables/Reconstruction_Output/dense/kernel/cond/switch_fIdentity>BackupVariables/Reconstruction_Output/dense/kernel/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
?BackupVariables/Reconstruction_Output/dense/kernel/cond/pred_idIdentityHBackupVariables/Reconstruction_Output/dense/kernel/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
<BackupVariables/Reconstruction_Output/dense/kernel/cond/readIdentityEBackupVariables/Reconstruction_Output/dense/kernel/cond/read/Switch:1"/device:CPU:0*
_output_shapes
:	?*
T0
?
CBackupVariables/Reconstruction_Output/dense/kernel/cond/read/Switch	RefSwitch"Reconstruction_Output/dense/kernel?BackupVariables/Reconstruction_Output/dense/kernel/cond/pred_id"/device:GPU:1*
T0**
_output_shapes
:	?:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
@BackupVariables/Reconstruction_Output/dense/kernel/cond/Switch_1Switch=Reconstruction_Output/dense/kernel/Initializer/random_uniform?BackupVariables/Reconstruction_Output/dense/kernel/cond/pred_id**
_output_shapes
:	?:	?*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
T0
?
=BackupVariables/Reconstruction_Output/dense/kernel/cond/MergeMerge@BackupVariables/Reconstruction_Output/dense/kernel/cond/Switch_1<BackupVariables/Reconstruction_Output/dense/kernel/cond/read"/device:CPU:0*
N*!
_output_shapes
:	?: *
T0
?
VBackupVariables/cond_14/read/Switch_BackupVariables/Reconstruction_Output/dense/kernelSwitch=BackupVariables/Reconstruction_Output/dense/kernel/cond/MergeBackupVariables/cond_14/pred_id"/device:CPU:0*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel**
_output_shapes
:	?:	?
?
OBackupVariables/cond_14/read_BackupVariables/Reconstruction_Output/dense/kernelIdentityXBackupVariables/cond_14/read/Switch_BackupVariables/Reconstruction_Output/dense/kernel:1"/device:CPU:0*
T0*
_output_shapes
:	?
?
PBackupVariables/cond_14/Merge_BackupVariables/Reconstruction_Output/dense/kernelMerge BackupVariables/cond_14/Switch_1OBackupVariables/cond_14/read_BackupVariables/Reconstruction_Output/dense/kernel"/device:CPU:0*
N*
T0*!
_output_shapes
:	?: 
?
9BackupVariables/Reconstruction_Output/dense/kernel/AssignAssign2BackupVariables/Reconstruction_Output/dense/kernelPBackupVariables/cond_14/Merge_BackupVariables/Reconstruction_Output/dense/kernel"/device:CPU:0*E
_class;
97loc:@BackupVariables/Reconstruction_Output/dense/kernel*
T0*
_output_shapes
:	?
?
7BackupVariables/Reconstruction_Output/dense/kernel/readIdentity2BackupVariables/Reconstruction_Output/dense/kernel"/device:CPU:0*
T0*E
_class;
97loc:@BackupVariables/Reconstruction_Output/dense/kernel*
_output_shapes
:	?
?
(BackupVariables/IsVariableInitialized_15IsVariableInitialized Reconstruction_Output/dense/bias"/device:GPU:1*
dtype0*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
: 
?
BackupVariables/cond_15/SwitchSwitch(BackupVariables/IsVariableInitialized_15(BackupVariables/IsVariableInitialized_15"/device:CPU:0*
T0
*
_output_shapes
: : 
~
 BackupVariables/cond_15/switch_tIdentity BackupVariables/cond_15/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
|
 BackupVariables/cond_15/switch_fIdentityBackupVariables/cond_15/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_15/pred_idIdentity(BackupVariables/IsVariableInitialized_15"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_15/readIdentity%BackupVariables/cond_15/read/Switch:1"/device:CPU:0*
_output_shapes
:*
T0
?
#BackupVariables/cond_15/read/Switch	RefSwitch Reconstruction_Output/dense/biasBackupVariables/cond_15/pred_id"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias* 
_output_shapes
::*
T0
?
 BackupVariables/cond_15/Switch_1Switch2Reconstruction_Output/dense/bias/Initializer/zerosBackupVariables/cond_15/pred_id*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
T0* 
_output_shapes
::
?
BackupVariables/cond_15/MergeMerge BackupVariables/cond_15/Switch_1BackupVariables/cond_15/read"/device:CPU:0*
T0*
_output_shapes

:: *
N
?
0BackupVariables/Reconstruction_Output/dense/bias
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:*
shape:
?
FBackupVariables/Reconstruction_Output/dense/bias/IsVariableInitializedIsVariableInitialized Reconstruction_Output/dense/bias"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
: *
dtype0
?
<BackupVariables/Reconstruction_Output/dense/bias/cond/SwitchSwitchFBackupVariables/Reconstruction_Output/dense/bias/IsVariableInitializedFBackupVariables/Reconstruction_Output/dense/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
>BackupVariables/Reconstruction_Output/dense/bias/cond/switch_tIdentity>BackupVariables/Reconstruction_Output/dense/bias/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
>BackupVariables/Reconstruction_Output/dense/bias/cond/switch_fIdentity<BackupVariables/Reconstruction_Output/dense/bias/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
=BackupVariables/Reconstruction_Output/dense/bias/cond/pred_idIdentityFBackupVariables/Reconstruction_Output/dense/bias/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
:BackupVariables/Reconstruction_Output/dense/bias/cond/readIdentityCBackupVariables/Reconstruction_Output/dense/bias/cond/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
:
?
ABackupVariables/Reconstruction_Output/dense/bias/cond/read/Switch	RefSwitch Reconstruction_Output/dense/bias=BackupVariables/Reconstruction_Output/dense/bias/cond/pred_id"/device:GPU:1*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias* 
_output_shapes
::
?
>BackupVariables/Reconstruction_Output/dense/bias/cond/Switch_1Switch2Reconstruction_Output/dense/bias/Initializer/zeros=BackupVariables/Reconstruction_Output/dense/bias/cond/pred_id* 
_output_shapes
::*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
T0
?
;BackupVariables/Reconstruction_Output/dense/bias/cond/MergeMerge>BackupVariables/Reconstruction_Output/dense/bias/cond/Switch_1:BackupVariables/Reconstruction_Output/dense/bias/cond/read"/device:CPU:0*
_output_shapes

:: *
T0*
N
?
TBackupVariables/cond_15/read/Switch_BackupVariables/Reconstruction_Output/dense/biasSwitch;BackupVariables/Reconstruction_Output/dense/bias/cond/MergeBackupVariables/cond_15/pred_id"/device:CPU:0* 
_output_shapes
::*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
MBackupVariables/cond_15/read_BackupVariables/Reconstruction_Output/dense/biasIdentityVBackupVariables/cond_15/read/Switch_BackupVariables/Reconstruction_Output/dense/bias:1"/device:CPU:0*
T0*
_output_shapes
:
?
NBackupVariables/cond_15/Merge_BackupVariables/Reconstruction_Output/dense/biasMerge BackupVariables/cond_15/Switch_1MBackupVariables/cond_15/read_BackupVariables/Reconstruction_Output/dense/bias"/device:CPU:0*
T0*
_output_shapes

:: *
N
?
7BackupVariables/Reconstruction_Output/dense/bias/AssignAssign0BackupVariables/Reconstruction_Output/dense/biasNBackupVariables/cond_15/Merge_BackupVariables/Reconstruction_Output/dense/bias"/device:CPU:0*
T0*
_output_shapes
:*C
_class9
75loc:@BackupVariables/Reconstruction_Output/dense/bias
?
5BackupVariables/Reconstruction_Output/dense/bias/readIdentity0BackupVariables/Reconstruction_Output/dense/bias"/device:CPU:0*
T0*
_output_shapes
:*C
_class9
75loc:@BackupVariables/Reconstruction_Output/dense/bias
?
(BackupVariables/IsVariableInitialized_16IsVariableInitializeddense/kernel"/device:GPU:1*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
?
BackupVariables/cond_16/SwitchSwitch(BackupVariables/IsVariableInitialized_16(BackupVariables/IsVariableInitialized_16"/device:CPU:0*
T0
*
_output_shapes
: : 
~
 BackupVariables/cond_16/switch_tIdentity BackupVariables/cond_16/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

|
 BackupVariables/cond_16/switch_fIdentityBackupVariables/cond_16/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_16/pred_idIdentity(BackupVariables/IsVariableInitialized_16"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_16/readIdentity%BackupVariables/cond_16/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
:	? 
?
#BackupVariables/cond_16/read/Switch	RefSwitchdense/kernelBackupVariables/cond_16/pred_id"/device:GPU:1**
_output_shapes
:	? :	? *
T0*
_class
loc:@dense/kernel
?
 BackupVariables/cond_16/Switch_1Switch'dense/kernel/Initializer/random_uniformBackupVariables/cond_16/pred_id*
T0**
_output_shapes
:	? :	? *
_class
loc:@dense/kernel
?
BackupVariables/cond_16/MergeMerge BackupVariables/cond_16/Switch_1BackupVariables/cond_16/read"/device:CPU:0*
T0*
N*!
_output_shapes
:	? : 
}
BackupVariables/dense/kernel
VariableV2"/device:CPU:0*
_output_shapes
:	? *
dtype0*
shape:	? 
?
2BackupVariables/dense/kernel/IsVariableInitializedIsVariableInitializeddense/kernel"/device:GPU:1*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: 
?
(BackupVariables/dense/kernel/cond/SwitchSwitch2BackupVariables/dense/kernel/IsVariableInitialized2BackupVariables/dense/kernel/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: : 
?
*BackupVariables/dense/kernel/cond/switch_tIdentity*BackupVariables/dense/kernel/cond/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
?
*BackupVariables/dense/kernel/cond/switch_fIdentity(BackupVariables/dense/kernel/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
)BackupVariables/dense/kernel/cond/pred_idIdentity2BackupVariables/dense/kernel/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
&BackupVariables/dense/kernel/cond/readIdentity/BackupVariables/dense/kernel/cond/read/Switch:1"/device:CPU:0*
_output_shapes
:	? *
T0
?
-BackupVariables/dense/kernel/cond/read/Switch	RefSwitchdense/kernel)BackupVariables/dense/kernel/cond/pred_id"/device:GPU:1*
_class
loc:@dense/kernel*
T0**
_output_shapes
:	? :	? 
?
*BackupVariables/dense/kernel/cond/Switch_1Switch'dense/kernel/Initializer/random_uniform)BackupVariables/dense/kernel/cond/pred_id**
_output_shapes
:	? :	? *
_class
loc:@dense/kernel*
T0
?
'BackupVariables/dense/kernel/cond/MergeMerge*BackupVariables/dense/kernel/cond/Switch_1&BackupVariables/dense/kernel/cond/read"/device:CPU:0*
T0*
N*!
_output_shapes
:	? : 
?
@BackupVariables/cond_16/read/Switch_BackupVariables/dense/kernelSwitch'BackupVariables/dense/kernel/cond/MergeBackupVariables/cond_16/pred_id"/device:CPU:0**
_output_shapes
:	? :	? *
T0*
_class
loc:@dense/kernel
?
9BackupVariables/cond_16/read_BackupVariables/dense/kernelIdentityBBackupVariables/cond_16/read/Switch_BackupVariables/dense/kernel:1"/device:CPU:0*
_output_shapes
:	? *
T0
?
:BackupVariables/cond_16/Merge_BackupVariables/dense/kernelMerge BackupVariables/cond_16/Switch_19BackupVariables/cond_16/read_BackupVariables/dense/kernel"/device:CPU:0*
N*!
_output_shapes
:	? : *
T0
?
#BackupVariables/dense/kernel/AssignAssignBackupVariables/dense/kernel:BackupVariables/cond_16/Merge_BackupVariables/dense/kernel"/device:CPU:0*
_output_shapes
:	? *
T0*/
_class%
#!loc:@BackupVariables/dense/kernel
?
!BackupVariables/dense/kernel/readIdentityBackupVariables/dense/kernel"/device:CPU:0*
_output_shapes
:	? *
T0*/
_class%
#!loc:@BackupVariables/dense/kernel
?
(BackupVariables/IsVariableInitialized_17IsVariableInitialized
dense/bias"/device:GPU:1*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
?
BackupVariables/cond_17/SwitchSwitch(BackupVariables/IsVariableInitialized_17(BackupVariables/IsVariableInitialized_17"/device:CPU:0*
_output_shapes
: : *
T0

~
 BackupVariables/cond_17/switch_tIdentity BackupVariables/cond_17/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
|
 BackupVariables/cond_17/switch_fIdentityBackupVariables/cond_17/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_17/pred_idIdentity(BackupVariables/IsVariableInitialized_17"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_17/readIdentity%BackupVariables/cond_17/read/Switch:1"/device:CPU:0*
_output_shapes
: *
T0
?
#BackupVariables/cond_17/read/Switch	RefSwitch
dense/biasBackupVariables/cond_17/pred_id"/device:GPU:1*
_class
loc:@dense/bias*
T0* 
_output_shapes
: : 
?
 BackupVariables/cond_17/Switch_1Switchdense/bias/Initializer/zerosBackupVariables/cond_17/pred_id*
T0*
_class
loc:@dense/bias* 
_output_shapes
: : 
?
BackupVariables/cond_17/MergeMerge BackupVariables/cond_17/Switch_1BackupVariables/cond_17/read"/device:CPU:0*
_output_shapes

: : *
T0*
N
q
BackupVariables/dense/bias
VariableV2"/device:CPU:0*
_output_shapes
: *
shape: *
dtype0
?
0BackupVariables/dense/bias/IsVariableInitializedIsVariableInitialized
dense/bias"/device:GPU:1*
_output_shapes
: *
dtype0*
_class
loc:@dense/bias
?
&BackupVariables/dense/bias/cond/SwitchSwitch0BackupVariables/dense/bias/IsVariableInitialized0BackupVariables/dense/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
(BackupVariables/dense/bias/cond/switch_tIdentity(BackupVariables/dense/bias/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
(BackupVariables/dense/bias/cond/switch_fIdentity&BackupVariables/dense/bias/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
'BackupVariables/dense/bias/cond/pred_idIdentity0BackupVariables/dense/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
$BackupVariables/dense/bias/cond/readIdentity-BackupVariables/dense/bias/cond/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
: 
?
+BackupVariables/dense/bias/cond/read/Switch	RefSwitch
dense/bias'BackupVariables/dense/bias/cond/pred_id"/device:GPU:1* 
_output_shapes
: : *
_class
loc:@dense/bias*
T0
?
(BackupVariables/dense/bias/cond/Switch_1Switchdense/bias/Initializer/zeros'BackupVariables/dense/bias/cond/pred_id*
_class
loc:@dense/bias* 
_output_shapes
: : *
T0
?
%BackupVariables/dense/bias/cond/MergeMerge(BackupVariables/dense/bias/cond/Switch_1$BackupVariables/dense/bias/cond/read"/device:CPU:0*
N*
_output_shapes

: : *
T0
?
>BackupVariables/cond_17/read/Switch_BackupVariables/dense/biasSwitch%BackupVariables/dense/bias/cond/MergeBackupVariables/cond_17/pred_id"/device:CPU:0*
T0*
_class
loc:@dense/bias* 
_output_shapes
: : 
?
7BackupVariables/cond_17/read_BackupVariables/dense/biasIdentity@BackupVariables/cond_17/read/Switch_BackupVariables/dense/bias:1"/device:CPU:0*
T0*
_output_shapes
: 
?
8BackupVariables/cond_17/Merge_BackupVariables/dense/biasMerge BackupVariables/cond_17/Switch_17BackupVariables/cond_17/read_BackupVariables/dense/bias"/device:CPU:0*
_output_shapes

: : *
T0*
N
?
!BackupVariables/dense/bias/AssignAssignBackupVariables/dense/bias8BackupVariables/cond_17/Merge_BackupVariables/dense/bias"/device:CPU:0*-
_class#
!loc:@BackupVariables/dense/bias*
T0*
_output_shapes
: 
?
BackupVariables/dense/bias/readIdentityBackupVariables/dense/bias"/device:CPU:0*
_output_shapes
: *-
_class#
!loc:@BackupVariables/dense/bias*
T0
?
(BackupVariables/IsVariableInitialized_18IsVariableInitializedFCU_muiltDense_x0/dense/kernel"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
dtype0*
_output_shapes
: 
?
BackupVariables/cond_18/SwitchSwitch(BackupVariables/IsVariableInitialized_18(BackupVariables/IsVariableInitialized_18"/device:CPU:0*
T0
*
_output_shapes
: : 
~
 BackupVariables/cond_18/switch_tIdentity BackupVariables/cond_18/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
|
 BackupVariables/cond_18/switch_fIdentityBackupVariables/cond_18/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_18/pred_idIdentity(BackupVariables/IsVariableInitialized_18"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_18/readIdentity%BackupVariables/cond_18/read/Switch:1"/device:CPU:0*
_output_shapes

:  *
T0
?
#BackupVariables/cond_18/read/Switch	RefSwitchFCU_muiltDense_x0/dense/kernelBackupVariables/cond_18/pred_id"/device:GPU:1*
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*(
_output_shapes
:  :  
?
 BackupVariables/cond_18/Switch_1Switch9FCU_muiltDense_x0/dense/kernel/Initializer/random_uniformBackupVariables/cond_18/pred_id*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*(
_output_shapes
:  :  *
T0
?
BackupVariables/cond_18/MergeMerge BackupVariables/cond_18/Switch_1BackupVariables/cond_18/read"/device:CPU:0* 
_output_shapes
:  : *
T0*
N
?
.BackupVariables/FCU_muiltDense_x0/dense/kernel
VariableV2"/device:CPU:0*
_output_shapes

:  *
shape
:  *
dtype0
?
DBackupVariables/FCU_muiltDense_x0/dense/kernel/IsVariableInitializedIsVariableInitializedFCU_muiltDense_x0/dense/kernel"/device:GPU:1*
dtype0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes
: 
?
:BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/SwitchSwitchDBackupVariables/FCU_muiltDense_x0/dense/kernel/IsVariableInitializedDBackupVariables/FCU_muiltDense_x0/dense/kernel/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
<BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/switch_tIdentity<BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
<BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/switch_fIdentity:BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
;BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/pred_idIdentityDBackupVariables/FCU_muiltDense_x0/dense/kernel/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
8BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/readIdentityABackupVariables/FCU_muiltDense_x0/dense/kernel/cond/read/Switch:1"/device:CPU:0*
_output_shapes

:  *
T0
?
?BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/read/Switch	RefSwitchFCU_muiltDense_x0/dense/kernel;BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/pred_id"/device:GPU:1*(
_output_shapes
:  :  *
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
<BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/Switch_1Switch9FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform;BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/pred_id*(
_output_shapes
:  :  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
T0
?
9BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/MergeMerge<BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/Switch_18BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/read"/device:CPU:0*
N* 
_output_shapes
:  : *
T0
?
RBackupVariables/cond_18/read/Switch_BackupVariables/FCU_muiltDense_x0/dense/kernelSwitch9BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/MergeBackupVariables/cond_18/pred_id"/device:CPU:0*
T0*(
_output_shapes
:  :  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
KBackupVariables/cond_18/read_BackupVariables/FCU_muiltDense_x0/dense/kernelIdentityTBackupVariables/cond_18/read/Switch_BackupVariables/FCU_muiltDense_x0/dense/kernel:1"/device:CPU:0*
T0*
_output_shapes

:  
?
LBackupVariables/cond_18/Merge_BackupVariables/FCU_muiltDense_x0/dense/kernelMerge BackupVariables/cond_18/Switch_1KBackupVariables/cond_18/read_BackupVariables/FCU_muiltDense_x0/dense/kernel"/device:CPU:0*
T0* 
_output_shapes
:  : *
N
?
5BackupVariables/FCU_muiltDense_x0/dense/kernel/AssignAssign.BackupVariables/FCU_muiltDense_x0/dense/kernelLBackupVariables/cond_18/Merge_BackupVariables/FCU_muiltDense_x0/dense/kernel"/device:CPU:0*
T0*A
_class7
53loc:@BackupVariables/FCU_muiltDense_x0/dense/kernel*
_output_shapes

:  
?
3BackupVariables/FCU_muiltDense_x0/dense/kernel/readIdentity.BackupVariables/FCU_muiltDense_x0/dense/kernel"/device:CPU:0*
T0*
_output_shapes

:  *A
_class7
53loc:@BackupVariables/FCU_muiltDense_x0/dense/kernel
?
(BackupVariables/IsVariableInitialized_19IsVariableInitializedFCU_muiltDense_x0/dense/bias"/device:GPU:1*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
dtype0
?
BackupVariables/cond_19/SwitchSwitch(BackupVariables/IsVariableInitialized_19(BackupVariables/IsVariableInitialized_19"/device:CPU:0*
T0
*
_output_shapes
: : 
~
 BackupVariables/cond_19/switch_tIdentity BackupVariables/cond_19/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

|
 BackupVariables/cond_19/switch_fIdentityBackupVariables/cond_19/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_19/pred_idIdentity(BackupVariables/IsVariableInitialized_19"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_19/readIdentity%BackupVariables/cond_19/read/Switch:1"/device:CPU:0*
_output_shapes
: *
T0
?
#BackupVariables/cond_19/read/Switch	RefSwitchFCU_muiltDense_x0/dense/biasBackupVariables/cond_19/pred_id"/device:GPU:1*
T0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias* 
_output_shapes
: : 
?
 BackupVariables/cond_19/Switch_1Switch.FCU_muiltDense_x0/dense/bias/Initializer/zerosBackupVariables/cond_19/pred_id*
T0* 
_output_shapes
: : */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
BackupVariables/cond_19/MergeMerge BackupVariables/cond_19/Switch_1BackupVariables/cond_19/read"/device:CPU:0*
T0*
_output_shapes

: : *
N
?
,BackupVariables/FCU_muiltDense_x0/dense/bias
VariableV2"/device:CPU:0*
shape: *
_output_shapes
: *
dtype0
?
BBackupVariables/FCU_muiltDense_x0/dense/bias/IsVariableInitializedIsVariableInitializedFCU_muiltDense_x0/dense/bias"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
_output_shapes
: *
dtype0
?
8BackupVariables/FCU_muiltDense_x0/dense/bias/cond/SwitchSwitchBBackupVariables/FCU_muiltDense_x0/dense/bias/IsVariableInitializedBBackupVariables/FCU_muiltDense_x0/dense/bias/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: : 
?
:BackupVariables/FCU_muiltDense_x0/dense/bias/cond/switch_tIdentity:BackupVariables/FCU_muiltDense_x0/dense/bias/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
:BackupVariables/FCU_muiltDense_x0/dense/bias/cond/switch_fIdentity8BackupVariables/FCU_muiltDense_x0/dense/bias/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
9BackupVariables/FCU_muiltDense_x0/dense/bias/cond/pred_idIdentityBBackupVariables/FCU_muiltDense_x0/dense/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: *
T0

?
6BackupVariables/FCU_muiltDense_x0/dense/bias/cond/readIdentity?BackupVariables/FCU_muiltDense_x0/dense/bias/cond/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
: 
?
=BackupVariables/FCU_muiltDense_x0/dense/bias/cond/read/Switch	RefSwitchFCU_muiltDense_x0/dense/bias9BackupVariables/FCU_muiltDense_x0/dense/bias/cond/pred_id"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0* 
_output_shapes
: : 
?
:BackupVariables/FCU_muiltDense_x0/dense/bias/cond/Switch_1Switch.FCU_muiltDense_x0/dense/bias/Initializer/zeros9BackupVariables/FCU_muiltDense_x0/dense/bias/cond/pred_id* 
_output_shapes
: : *
T0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
7BackupVariables/FCU_muiltDense_x0/dense/bias/cond/MergeMerge:BackupVariables/FCU_muiltDense_x0/dense/bias/cond/Switch_16BackupVariables/FCU_muiltDense_x0/dense/bias/cond/read"/device:CPU:0*
_output_shapes

: : *
T0*
N
?
PBackupVariables/cond_19/read/Switch_BackupVariables/FCU_muiltDense_x0/dense/biasSwitch7BackupVariables/FCU_muiltDense_x0/dense/bias/cond/MergeBackupVariables/cond_19/pred_id"/device:CPU:0* 
_output_shapes
: : */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0
?
IBackupVariables/cond_19/read_BackupVariables/FCU_muiltDense_x0/dense/biasIdentityRBackupVariables/cond_19/read/Switch_BackupVariables/FCU_muiltDense_x0/dense/bias:1"/device:CPU:0*
T0*
_output_shapes
: 
?
JBackupVariables/cond_19/Merge_BackupVariables/FCU_muiltDense_x0/dense/biasMerge BackupVariables/cond_19/Switch_1IBackupVariables/cond_19/read_BackupVariables/FCU_muiltDense_x0/dense/bias"/device:CPU:0*
T0*
_output_shapes

: : *
N
?
3BackupVariables/FCU_muiltDense_x0/dense/bias/AssignAssign,BackupVariables/FCU_muiltDense_x0/dense/biasJBackupVariables/cond_19/Merge_BackupVariables/FCU_muiltDense_x0/dense/bias"/device:CPU:0*
_output_shapes
: *?
_class5
31loc:@BackupVariables/FCU_muiltDense_x0/dense/bias*
T0
?
1BackupVariables/FCU_muiltDense_x0/dense/bias/readIdentity,BackupVariables/FCU_muiltDense_x0/dense/bias"/device:CPU:0*
_output_shapes
: *
T0*?
_class5
31loc:@BackupVariables/FCU_muiltDense_x0/dense/bias
?
(BackupVariables/IsVariableInitialized_20IsVariableInitializedFCU_muiltDense_x0/beta"/device:GPU:1*
dtype0*
_output_shapes
: *)
_class
loc:@FCU_muiltDense_x0/beta
?
BackupVariables/cond_20/SwitchSwitch(BackupVariables/IsVariableInitialized_20(BackupVariables/IsVariableInitialized_20"/device:CPU:0*
T0
*
_output_shapes
: : 
~
 BackupVariables/cond_20/switch_tIdentity BackupVariables/cond_20/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
|
 BackupVariables/cond_20/switch_fIdentityBackupVariables/cond_20/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_20/pred_idIdentity(BackupVariables/IsVariableInitialized_20"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_20/readIdentity%BackupVariables/cond_20/read/Switch:1"/device:CPU:0*
_output_shapes
: *
T0
?
#BackupVariables/cond_20/read/Switch	RefSwitchFCU_muiltDense_x0/betaBackupVariables/cond_20/pred_id"/device:GPU:1* 
_output_shapes
: : *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
 BackupVariables/cond_20/Switch_1Switch(FCU_muiltDense_x0/beta/Initializer/zerosBackupVariables/cond_20/pred_id* 
_output_shapes
: : *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
BackupVariables/cond_20/MergeMerge BackupVariables/cond_20/Switch_1BackupVariables/cond_20/read"/device:CPU:0*
N*
T0*
_output_shapes

: : 
}
&BackupVariables/FCU_muiltDense_x0/beta
VariableV2"/device:CPU:0*
_output_shapes
: *
shape: *
dtype0
?
<BackupVariables/FCU_muiltDense_x0/beta/IsVariableInitializedIsVariableInitializedFCU_muiltDense_x0/beta"/device:GPU:1*
_output_shapes
: *)
_class
loc:@FCU_muiltDense_x0/beta*
dtype0
?
2BackupVariables/FCU_muiltDense_x0/beta/cond/SwitchSwitch<BackupVariables/FCU_muiltDense_x0/beta/IsVariableInitialized<BackupVariables/FCU_muiltDense_x0/beta/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: : 
?
4BackupVariables/FCU_muiltDense_x0/beta/cond/switch_tIdentity4BackupVariables/FCU_muiltDense_x0/beta/cond/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
?
4BackupVariables/FCU_muiltDense_x0/beta/cond/switch_fIdentity2BackupVariables/FCU_muiltDense_x0/beta/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
3BackupVariables/FCU_muiltDense_x0/beta/cond/pred_idIdentity<BackupVariables/FCU_muiltDense_x0/beta/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
0BackupVariables/FCU_muiltDense_x0/beta/cond/readIdentity9BackupVariables/FCU_muiltDense_x0/beta/cond/read/Switch:1"/device:CPU:0*
_output_shapes
: *
T0
?
7BackupVariables/FCU_muiltDense_x0/beta/cond/read/Switch	RefSwitchFCU_muiltDense_x0/beta3BackupVariables/FCU_muiltDense_x0/beta/cond/pred_id"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta* 
_output_shapes
: : *
T0
?
4BackupVariables/FCU_muiltDense_x0/beta/cond/Switch_1Switch(FCU_muiltDense_x0/beta/Initializer/zeros3BackupVariables/FCU_muiltDense_x0/beta/cond/pred_id* 
_output_shapes
: : *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
1BackupVariables/FCU_muiltDense_x0/beta/cond/MergeMerge4BackupVariables/FCU_muiltDense_x0/beta/cond/Switch_10BackupVariables/FCU_muiltDense_x0/beta/cond/read"/device:CPU:0*
T0*
N*
_output_shapes

: : 
?
JBackupVariables/cond_20/read/Switch_BackupVariables/FCU_muiltDense_x0/betaSwitch1BackupVariables/FCU_muiltDense_x0/beta/cond/MergeBackupVariables/cond_20/pred_id"/device:CPU:0*)
_class
loc:@FCU_muiltDense_x0/beta*
T0* 
_output_shapes
: : 
?
CBackupVariables/cond_20/read_BackupVariables/FCU_muiltDense_x0/betaIdentityLBackupVariables/cond_20/read/Switch_BackupVariables/FCU_muiltDense_x0/beta:1"/device:CPU:0*
_output_shapes
: *
T0
?
DBackupVariables/cond_20/Merge_BackupVariables/FCU_muiltDense_x0/betaMerge BackupVariables/cond_20/Switch_1CBackupVariables/cond_20/read_BackupVariables/FCU_muiltDense_x0/beta"/device:CPU:0*
T0*
_output_shapes

: : *
N
?
-BackupVariables/FCU_muiltDense_x0/beta/AssignAssign&BackupVariables/FCU_muiltDense_x0/betaDBackupVariables/cond_20/Merge_BackupVariables/FCU_muiltDense_x0/beta"/device:CPU:0*
_output_shapes
: *9
_class/
-+loc:@BackupVariables/FCU_muiltDense_x0/beta*
T0
?
+BackupVariables/FCU_muiltDense_x0/beta/readIdentity&BackupVariables/FCU_muiltDense_x0/beta"/device:CPU:0*
_output_shapes
: *
T0*9
_class/
-+loc:@BackupVariables/FCU_muiltDense_x0/beta
?
(BackupVariables/IsVariableInitialized_21IsVariableInitializedFCU_muiltDense_x0/gamma"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: *
dtype0
?
BackupVariables/cond_21/SwitchSwitch(BackupVariables/IsVariableInitialized_21(BackupVariables/IsVariableInitialized_21"/device:CPU:0*
_output_shapes
: : *
T0

~
 BackupVariables/cond_21/switch_tIdentity BackupVariables/cond_21/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

|
 BackupVariables/cond_21/switch_fIdentityBackupVariables/cond_21/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_21/pred_idIdentity(BackupVariables/IsVariableInitialized_21"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_21/readIdentity%BackupVariables/cond_21/read/Switch:1"/device:CPU:0*
_output_shapes
: *
T0
?
#BackupVariables/cond_21/read/Switch	RefSwitchFCU_muiltDense_x0/gammaBackupVariables/cond_21/pred_id"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
T0* 
_output_shapes
: : 
?
 BackupVariables/cond_21/Switch_1Switch(FCU_muiltDense_x0/gamma/Initializer/onesBackupVariables/cond_21/pred_id**
_class 
loc:@FCU_muiltDense_x0/gamma* 
_output_shapes
: : *
T0
?
BackupVariables/cond_21/MergeMerge BackupVariables/cond_21/Switch_1BackupVariables/cond_21/read"/device:CPU:0*
N*
_output_shapes

: : *
T0
~
'BackupVariables/FCU_muiltDense_x0/gamma
VariableV2"/device:CPU:0*
_output_shapes
: *
dtype0*
shape: 
?
=BackupVariables/FCU_muiltDense_x0/gamma/IsVariableInitializedIsVariableInitializedFCU_muiltDense_x0/gamma"/device:GPU:1*
_output_shapes
: *
dtype0**
_class 
loc:@FCU_muiltDense_x0/gamma
?
3BackupVariables/FCU_muiltDense_x0/gamma/cond/SwitchSwitch=BackupVariables/FCU_muiltDense_x0/gamma/IsVariableInitialized=BackupVariables/FCU_muiltDense_x0/gamma/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
5BackupVariables/FCU_muiltDense_x0/gamma/cond/switch_tIdentity5BackupVariables/FCU_muiltDense_x0/gamma/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
5BackupVariables/FCU_muiltDense_x0/gamma/cond/switch_fIdentity3BackupVariables/FCU_muiltDense_x0/gamma/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
4BackupVariables/FCU_muiltDense_x0/gamma/cond/pred_idIdentity=BackupVariables/FCU_muiltDense_x0/gamma/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
1BackupVariables/FCU_muiltDense_x0/gamma/cond/readIdentity:BackupVariables/FCU_muiltDense_x0/gamma/cond/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
: 
?
8BackupVariables/FCU_muiltDense_x0/gamma/cond/read/Switch	RefSwitchFCU_muiltDense_x0/gamma4BackupVariables/FCU_muiltDense_x0/gamma/cond/pred_id"/device:GPU:1*
T0**
_class 
loc:@FCU_muiltDense_x0/gamma* 
_output_shapes
: : 
?
5BackupVariables/FCU_muiltDense_x0/gamma/cond/Switch_1Switch(FCU_muiltDense_x0/gamma/Initializer/ones4BackupVariables/FCU_muiltDense_x0/gamma/cond/pred_id*
T0**
_class 
loc:@FCU_muiltDense_x0/gamma* 
_output_shapes
: : 
?
2BackupVariables/FCU_muiltDense_x0/gamma/cond/MergeMerge5BackupVariables/FCU_muiltDense_x0/gamma/cond/Switch_11BackupVariables/FCU_muiltDense_x0/gamma/cond/read"/device:CPU:0*
_output_shapes

: : *
N*
T0
?
KBackupVariables/cond_21/read/Switch_BackupVariables/FCU_muiltDense_x0/gammaSwitch2BackupVariables/FCU_muiltDense_x0/gamma/cond/MergeBackupVariables/cond_21/pred_id"/device:CPU:0**
_class 
loc:@FCU_muiltDense_x0/gamma* 
_output_shapes
: : *
T0
?
DBackupVariables/cond_21/read_BackupVariables/FCU_muiltDense_x0/gammaIdentityMBackupVariables/cond_21/read/Switch_BackupVariables/FCU_muiltDense_x0/gamma:1"/device:CPU:0*
_output_shapes
: *
T0
?
EBackupVariables/cond_21/Merge_BackupVariables/FCU_muiltDense_x0/gammaMerge BackupVariables/cond_21/Switch_1DBackupVariables/cond_21/read_BackupVariables/FCU_muiltDense_x0/gamma"/device:CPU:0*
_output_shapes

: : *
T0*
N
?
.BackupVariables/FCU_muiltDense_x0/gamma/AssignAssign'BackupVariables/FCU_muiltDense_x0/gammaEBackupVariables/cond_21/Merge_BackupVariables/FCU_muiltDense_x0/gamma"/device:CPU:0*:
_class0
.,loc:@BackupVariables/FCU_muiltDense_x0/gamma*
T0*
_output_shapes
: 
?
,BackupVariables/FCU_muiltDense_x0/gamma/readIdentity'BackupVariables/FCU_muiltDense_x0/gamma"/device:CPU:0*
T0*:
_class0
.,loc:@BackupVariables/FCU_muiltDense_x0/gamma*
_output_shapes
: 
?
(BackupVariables/IsVariableInitialized_22IsVariableInitializedOutput_/dense/kernel"/device:GPU:1*
_output_shapes
: *'
_class
loc:@Output_/dense/kernel*
dtype0
?
BackupVariables/cond_22/SwitchSwitch(BackupVariables/IsVariableInitialized_22(BackupVariables/IsVariableInitialized_22"/device:CPU:0*
_output_shapes
: : *
T0

~
 BackupVariables/cond_22/switch_tIdentity BackupVariables/cond_22/Switch:1"/device:CPU:0*
T0
*
_output_shapes
: 
|
 BackupVariables/cond_22/switch_fIdentityBackupVariables/cond_22/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_22/pred_idIdentity(BackupVariables/IsVariableInitialized_22"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_22/readIdentity%BackupVariables/cond_22/read/Switch:1"/device:CPU:0*
_output_shapes

: *
T0
?
#BackupVariables/cond_22/read/Switch	RefSwitchOutput_/dense/kernelBackupVariables/cond_22/pred_id"/device:GPU:1*
T0*(
_output_shapes
: : *'
_class
loc:@Output_/dense/kernel
?
 BackupVariables/cond_22/Switch_1Switch/Output_/dense/kernel/Initializer/random_uniformBackupVariables/cond_22/pred_id*
T0*'
_class
loc:@Output_/dense/kernel*(
_output_shapes
: : 
?
BackupVariables/cond_22/MergeMerge BackupVariables/cond_22/Switch_1BackupVariables/cond_22/read"/device:CPU:0*
N*
T0* 
_output_shapes
: : 
?
$BackupVariables/Output_/dense/kernel
VariableV2"/device:CPU:0*
dtype0*
shape
: *
_output_shapes

: 
?
:BackupVariables/Output_/dense/kernel/IsVariableInitializedIsVariableInitializedOutput_/dense/kernel"/device:GPU:1*
_output_shapes
: *'
_class
loc:@Output_/dense/kernel*
dtype0
?
0BackupVariables/Output_/dense/kernel/cond/SwitchSwitch:BackupVariables/Output_/dense/kernel/IsVariableInitialized:BackupVariables/Output_/dense/kernel/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
2BackupVariables/Output_/dense/kernel/cond/switch_tIdentity2BackupVariables/Output_/dense/kernel/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
2BackupVariables/Output_/dense/kernel/cond/switch_fIdentity0BackupVariables/Output_/dense/kernel/cond/Switch"/device:CPU:0*
_output_shapes
: *
T0

?
1BackupVariables/Output_/dense/kernel/cond/pred_idIdentity:BackupVariables/Output_/dense/kernel/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
.BackupVariables/Output_/dense/kernel/cond/readIdentity7BackupVariables/Output_/dense/kernel/cond/read/Switch:1"/device:CPU:0*
T0*
_output_shapes

: 
?
5BackupVariables/Output_/dense/kernel/cond/read/Switch	RefSwitchOutput_/dense/kernel1BackupVariables/Output_/dense/kernel/cond/pred_id"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*(
_output_shapes
: : *
T0
?
2BackupVariables/Output_/dense/kernel/cond/Switch_1Switch/Output_/dense/kernel/Initializer/random_uniform1BackupVariables/Output_/dense/kernel/cond/pred_id*
T0*(
_output_shapes
: : *'
_class
loc:@Output_/dense/kernel
?
/BackupVariables/Output_/dense/kernel/cond/MergeMerge2BackupVariables/Output_/dense/kernel/cond/Switch_1.BackupVariables/Output_/dense/kernel/cond/read"/device:CPU:0* 
_output_shapes
: : *
N*
T0
?
HBackupVariables/cond_22/read/Switch_BackupVariables/Output_/dense/kernelSwitch/BackupVariables/Output_/dense/kernel/cond/MergeBackupVariables/cond_22/pred_id"/device:CPU:0*(
_output_shapes
: : *
T0*'
_class
loc:@Output_/dense/kernel
?
ABackupVariables/cond_22/read_BackupVariables/Output_/dense/kernelIdentityJBackupVariables/cond_22/read/Switch_BackupVariables/Output_/dense/kernel:1"/device:CPU:0*
_output_shapes

: *
T0
?
BBackupVariables/cond_22/Merge_BackupVariables/Output_/dense/kernelMerge BackupVariables/cond_22/Switch_1ABackupVariables/cond_22/read_BackupVariables/Output_/dense/kernel"/device:CPU:0*
T0*
N* 
_output_shapes
: : 
?
+BackupVariables/Output_/dense/kernel/AssignAssign$BackupVariables/Output_/dense/kernelBBackupVariables/cond_22/Merge_BackupVariables/Output_/dense/kernel"/device:CPU:0*7
_class-
+)loc:@BackupVariables/Output_/dense/kernel*
T0*
_output_shapes

: 
?
)BackupVariables/Output_/dense/kernel/readIdentity$BackupVariables/Output_/dense/kernel"/device:CPU:0*
T0*
_output_shapes

: *7
_class-
+)loc:@BackupVariables/Output_/dense/kernel
?
(BackupVariables/IsVariableInitialized_23IsVariableInitializedOutput_/dense/bias"/device:GPU:1*
_output_shapes
: *%
_class
loc:@Output_/dense/bias*
dtype0
?
BackupVariables/cond_23/SwitchSwitch(BackupVariables/IsVariableInitialized_23(BackupVariables/IsVariableInitialized_23"/device:CPU:0*
T0
*
_output_shapes
: : 
~
 BackupVariables/cond_23/switch_tIdentity BackupVariables/cond_23/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

|
 BackupVariables/cond_23/switch_fIdentityBackupVariables/cond_23/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
BackupVariables/cond_23/pred_idIdentity(BackupVariables/IsVariableInitialized_23"/device:CPU:0*
_output_shapes
: *
T0

?
BackupVariables/cond_23/readIdentity%BackupVariables/cond_23/read/Switch:1"/device:CPU:0*
_output_shapes
:*
T0
?
#BackupVariables/cond_23/read/Switch	RefSwitchOutput_/dense/biasBackupVariables/cond_23/pred_id"/device:GPU:1*%
_class
loc:@Output_/dense/bias* 
_output_shapes
::*
T0
?
 BackupVariables/cond_23/Switch_1Switch$Output_/dense/bias/Initializer/zerosBackupVariables/cond_23/pred_id*
T0* 
_output_shapes
::*%
_class
loc:@Output_/dense/bias
?
BackupVariables/cond_23/MergeMerge BackupVariables/cond_23/Switch_1BackupVariables/cond_23/read"/device:CPU:0*
T0*
N*
_output_shapes

:: 
y
"BackupVariables/Output_/dense/bias
VariableV2"/device:CPU:0*
dtype0*
shape:*
_output_shapes
:
?
8BackupVariables/Output_/dense/bias/IsVariableInitializedIsVariableInitializedOutput_/dense/bias"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
dtype0*
_output_shapes
: 
?
.BackupVariables/Output_/dense/bias/cond/SwitchSwitch8BackupVariables/Output_/dense/bias/IsVariableInitialized8BackupVariables/Output_/dense/bias/IsVariableInitialized"/device:CPU:0*
_output_shapes
: : *
T0

?
0BackupVariables/Output_/dense/bias/cond/switch_tIdentity0BackupVariables/Output_/dense/bias/cond/Switch:1"/device:CPU:0*
_output_shapes
: *
T0

?
0BackupVariables/Output_/dense/bias/cond/switch_fIdentity.BackupVariables/Output_/dense/bias/cond/Switch"/device:CPU:0*
T0
*
_output_shapes
: 
?
/BackupVariables/Output_/dense/bias/cond/pred_idIdentity8BackupVariables/Output_/dense/bias/IsVariableInitialized"/device:CPU:0*
T0
*
_output_shapes
: 
?
,BackupVariables/Output_/dense/bias/cond/readIdentity5BackupVariables/Output_/dense/bias/cond/read/Switch:1"/device:CPU:0*
T0*
_output_shapes
:
?
3BackupVariables/Output_/dense/bias/cond/read/Switch	RefSwitchOutput_/dense/bias/BackupVariables/Output_/dense/bias/cond/pred_id"/device:GPU:1*
T0* 
_output_shapes
::*%
_class
loc:@Output_/dense/bias
?
0BackupVariables/Output_/dense/bias/cond/Switch_1Switch$Output_/dense/bias/Initializer/zeros/BackupVariables/Output_/dense/bias/cond/pred_id*%
_class
loc:@Output_/dense/bias* 
_output_shapes
::*
T0
?
-BackupVariables/Output_/dense/bias/cond/MergeMerge0BackupVariables/Output_/dense/bias/cond/Switch_1,BackupVariables/Output_/dense/bias/cond/read"/device:CPU:0*
_output_shapes

:: *
N*
T0
?
FBackupVariables/cond_23/read/Switch_BackupVariables/Output_/dense/biasSwitch-BackupVariables/Output_/dense/bias/cond/MergeBackupVariables/cond_23/pred_id"/device:CPU:0* 
_output_shapes
::*%
_class
loc:@Output_/dense/bias*
T0
?
?BackupVariables/cond_23/read_BackupVariables/Output_/dense/biasIdentityHBackupVariables/cond_23/read/Switch_BackupVariables/Output_/dense/bias:1"/device:CPU:0*
T0*
_output_shapes
:
?
@BackupVariables/cond_23/Merge_BackupVariables/Output_/dense/biasMerge BackupVariables/cond_23/Switch_1?BackupVariables/cond_23/read_BackupVariables/Output_/dense/bias"/device:CPU:0*
N*
T0*
_output_shapes

:: 
?
)BackupVariables/Output_/dense/bias/AssignAssign"BackupVariables/Output_/dense/bias@BackupVariables/cond_23/Merge_BackupVariables/Output_/dense/bias"/device:CPU:0*5
_class+
)'loc:@BackupVariables/Output_/dense/bias*
T0*
_output_shapes
:
?
'BackupVariables/Output_/dense/bias/readIdentity"BackupVariables/Output_/dense/bias"/device:CPU:0*
T0*
_output_shapes
:*5
_class+
)'loc:@BackupVariables/Output_/dense/bias
?
readIdentity4CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage"/device:CPU:0*
T0*#
_output_shapes
:?
?
AssignAssignCCN_1Conv_x0/convA10/kernelread"/device:GPU:1*#
_output_shapes
:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0
{
read_1Identity2CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage"/device:CPU:0*
_output_shapes	
:?*
T0
?
Assign_1AssignCCN_1Conv_x0/convA10/biasread_1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0*
_output_shapes	
:?
?
read_2Identity4CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage"/device:CPU:0*$
_output_shapes
:??*
T0
?
Assign_2AssignCCN_1Conv_x0/convB10/kernelread_2"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0
{
read_3Identity2CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage"/device:CPU:0*
T0*
_output_shapes	
:?
?
Assign_3AssignCCN_1Conv_x0/convB10/biasread_3"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
?
read_4Identity4CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage"/device:CPU:0*
T0*$
_output_shapes
:??
?
Assign_4AssignCCN_1Conv_x0/convB20/kernelread_4"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??
{
read_5Identity2CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage"/device:CPU:0*
_output_shapes	
:?*
T0
?
Assign_5AssignCCN_1Conv_x0/convB20/biasread_5"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
read_6Identity4CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage"/device:CPU:0*
T0*$
_output_shapes
:??
?
Assign_6AssignCCN_1Conv_x0/convA11/kernelread_6"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??
{
read_7Identity2CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage"/device:CPU:0*
_output_shapes	
:?*
T0
?
Assign_7AssignCCN_1Conv_x0/convA11/biasread_7"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias
?
read_8Identity4CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage"/device:CPU:0*$
_output_shapes
:??*
T0
?
Assign_8AssignCCN_1Conv_x0/convB11/kernelread_8"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0
{
read_9Identity2CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage"/device:CPU:0*
T0*
_output_shapes	
:?
?
Assign_9AssignCCN_1Conv_x0/convB11/biasread_9"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?*
T0
?
read_10Identity4CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage"/device:CPU:0*$
_output_shapes
:??*
T0
?
	Assign_10AssignCCN_1Conv_x0/convB21/kernelread_10"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??
|
read_11Identity2CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage"/device:CPU:0*
T0*
_output_shapes	
:?
?
	Assign_11AssignCCN_1Conv_x0/convB21/biasread_11"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
r
read_12Identity(Conv_out__/beta/ExponentialMovingAverage"/device:CPU:0*
T0*
_output_shapes	
:?
?
	Assign_12AssignConv_out__/betaread_12"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
T0*
_output_shapes	
:?
s
read_13Identity)Conv_out__/gamma/ExponentialMovingAverage"/device:CPU:0*
T0*
_output_shapes	
:?
?
	Assign_13AssignConv_out__/gammaread_13"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
_output_shapes	
:?*
T0
?
read_14Identity;Reconstruction_Output/dense/kernel/ExponentialMovingAverage"/device:CPU:0*
_output_shapes
:	?*
T0
?
	Assign_14Assign"Reconstruction_Output/dense/kernelread_14"/device:GPU:1*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?
?
read_15Identity9Reconstruction_Output/dense/bias/ExponentialMovingAverage"/device:CPU:0*
T0*
_output_shapes
:
?
	Assign_15Assign Reconstruction_Output/dense/biasread_15"/device:GPU:1*
_output_shapes
:*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias
s
read_16Identity%dense/kernel/ExponentialMovingAverage"/device:CPU:0*
_output_shapes
:	? *
T0
?
	Assign_16Assigndense/kernelread_16"/device:GPU:1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	? 
l
read_17Identity#dense/bias/ExponentialMovingAverage"/device:CPU:0*
_output_shapes
: *
T0
{
	Assign_17Assign
dense/biasread_17"/device:GPU:1*
_output_shapes
: *
T0*
_class
loc:@dense/bias
?
read_18Identity7FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage"/device:CPU:0*
T0*
_output_shapes

:  
?
	Assign_18AssignFCU_muiltDense_x0/dense/kernelread_18"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
T0*
_output_shapes

:  
~
read_19Identity5FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage"/device:CPU:0*
T0*
_output_shapes
: 
?
	Assign_19AssignFCU_muiltDense_x0/dense/biasread_19"/device:GPU:1*
_output_shapes
: *
T0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
x
read_20Identity/FCU_muiltDense_x0/beta/ExponentialMovingAverage"/device:CPU:0*
_output_shapes
: *
T0
?
	Assign_20AssignFCU_muiltDense_x0/betaread_20"/device:GPU:1*
T0*
_output_shapes
: *)
_class
loc:@FCU_muiltDense_x0/beta
y
read_21Identity0FCU_muiltDense_x0/gamma/ExponentialMovingAverage"/device:CPU:0*
T0*
_output_shapes
: 
?
	Assign_21AssignFCU_muiltDense_x0/gammaread_21"/device:GPU:1*
T0*
_output_shapes
: **
_class 
loc:@FCU_muiltDense_x0/gamma
z
read_22Identity-Output_/dense/kernel/ExponentialMovingAverage"/device:CPU:0*
_output_shapes

: *
T0
?
	Assign_22AssignOutput_/dense/kernelread_22"/device:GPU:1*
_output_shapes

: *
T0*'
_class
loc:@Output_/dense/kernel
t
read_23Identity+Output_/dense/bias/ExponentialMovingAverage"/device:CPU:0*
T0*
_output_shapes
:
?
	Assign_23AssignOutput_/dense/biasread_23"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
T0*
_output_shapes
:
?
group_deps_1NoOp^Assign	^Assign_1
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19	^Assign_2
^Assign_20
^Assign_21
^Assign_22
^Assign_23	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9"/device:GPU:1
m
read_24IdentityCCN_1Conv_x0/convA10/kernel"/device:CPU:0*
T0*#
_output_shapes
:?
?
	Assign_24Assign+BackupVariables/CCN_1Conv_x0/convA10/kernelread_24"/device:CPU:0*#
_output_shapes
:?*
T0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convA10/kernel
c
read_25IdentityCCN_1Conv_x0/convA10/bias"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_25Assign)BackupVariables/CCN_1Conv_x0/convA10/biasread_25"/device:CPU:0*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?
n
read_26IdentityCCN_1Conv_x0/convB10/kernel"/device:CPU:0*$
_output_shapes
:??*
T0
?
	Assign_26Assign+BackupVariables/CCN_1Conv_x0/convB10/kernelread_26"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??*
T0
c
read_27IdentityCCN_1Conv_x0/convB10/bias"/device:CPU:0*
T0*
_output_shapes	
:?
?
	Assign_27Assign)BackupVariables/CCN_1Conv_x0/convB10/biasread_27"/device:CPU:0*
_output_shapes	
:?*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB10/bias
n
read_28IdentityCCN_1Conv_x0/convB20/kernel"/device:CPU:0*$
_output_shapes
:??*
T0
?
	Assign_28Assign+BackupVariables/CCN_1Conv_x0/convB20/kernelread_28"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB20/kernel*
T0*$
_output_shapes
:??
c
read_29IdentityCCN_1Conv_x0/convB20/bias"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_29Assign)BackupVariables/CCN_1Conv_x0/convB20/biasread_29"/device:CPU:0*
_output_shapes	
:?*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB20/bias*
T0
n
read_30IdentityCCN_1Conv_x0/convA11/kernel"/device:CPU:0*
T0*$
_output_shapes
:??
?
	Assign_30Assign+BackupVariables/CCN_1Conv_x0/convA11/kernelread_30"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convA11/kernel*
T0*$
_output_shapes
:??
c
read_31IdentityCCN_1Conv_x0/convA11/bias"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_31Assign)BackupVariables/CCN_1Conv_x0/convA11/biasread_31"/device:CPU:0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?*
T0
n
read_32IdentityCCN_1Conv_x0/convB11/kernel"/device:CPU:0*
T0*$
_output_shapes
:??
?
	Assign_32Assign+BackupVariables/CCN_1Conv_x0/convB11/kernelread_32"/device:CPU:0*
T0*$
_output_shapes
:??*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB11/kernel
c
read_33IdentityCCN_1Conv_x0/convB11/bias"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_33Assign)BackupVariables/CCN_1Conv_x0/convB11/biasread_33"/device:CPU:0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?*
T0
n
read_34IdentityCCN_1Conv_x0/convB21/kernel"/device:CPU:0*
T0*$
_output_shapes
:??
?
	Assign_34Assign+BackupVariables/CCN_1Conv_x0/convB21/kernelread_34"/device:CPU:0*$
_output_shapes
:??*
T0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB21/kernel
c
read_35IdentityCCN_1Conv_x0/convB21/bias"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_35Assign)BackupVariables/CCN_1Conv_x0/convB21/biasread_35"/device:CPU:0*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB21/bias*
_output_shapes	
:?
Y
read_36IdentityConv_out__/beta"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_36AssignBackupVariables/Conv_out__/betaread_36"/device:CPU:0*
T0*
_output_shapes	
:?*2
_class(
&$loc:@BackupVariables/Conv_out__/beta
Z
read_37IdentityConv_out__/gamma"/device:CPU:0*
T0*
_output_shapes	
:?
?
	Assign_37Assign BackupVariables/Conv_out__/gammaread_37"/device:CPU:0*
T0*
_output_shapes	
:?*3
_class)
'%loc:@BackupVariables/Conv_out__/gamma
p
read_38Identity"Reconstruction_Output/dense/kernel"/device:CPU:0*
_output_shapes
:	?*
T0
?
	Assign_38Assign2BackupVariables/Reconstruction_Output/dense/kernelread_38"/device:CPU:0*E
_class;
97loc:@BackupVariables/Reconstruction_Output/dense/kernel*
_output_shapes
:	?*
T0
i
read_39Identity Reconstruction_Output/dense/bias"/device:CPU:0*
_output_shapes
:*
T0
?
	Assign_39Assign0BackupVariables/Reconstruction_Output/dense/biasread_39"/device:CPU:0*C
_class9
75loc:@BackupVariables/Reconstruction_Output/dense/bias*
T0*
_output_shapes
:
Z
read_40Identitydense/kernel"/device:CPU:0*
_output_shapes
:	? *
T0
?
	Assign_40AssignBackupVariables/dense/kernelread_40"/device:CPU:0*
T0*/
_class%
#!loc:@BackupVariables/dense/kernel*
_output_shapes
:	? 
S
read_41Identity
dense/bias"/device:CPU:0*
_output_shapes
: *
T0
?
	Assign_41AssignBackupVariables/dense/biasread_41"/device:CPU:0*-
_class#
!loc:@BackupVariables/dense/bias*
_output_shapes
: *
T0
k
read_42IdentityFCU_muiltDense_x0/dense/kernel"/device:CPU:0*
T0*
_output_shapes

:  
?
	Assign_42Assign.BackupVariables/FCU_muiltDense_x0/dense/kernelread_42"/device:CPU:0*
T0*A
_class7
53loc:@BackupVariables/FCU_muiltDense_x0/dense/kernel*
_output_shapes

:  
e
read_43IdentityFCU_muiltDense_x0/dense/bias"/device:CPU:0*
_output_shapes
: *
T0
?
	Assign_43Assign,BackupVariables/FCU_muiltDense_x0/dense/biasread_43"/device:CPU:0*
_output_shapes
: *?
_class5
31loc:@BackupVariables/FCU_muiltDense_x0/dense/bias*
T0
_
read_44IdentityFCU_muiltDense_x0/beta"/device:CPU:0*
T0*
_output_shapes
: 
?
	Assign_44Assign&BackupVariables/FCU_muiltDense_x0/betaread_44"/device:CPU:0*
T0*
_output_shapes
: *9
_class/
-+loc:@BackupVariables/FCU_muiltDense_x0/beta
`
read_45IdentityFCU_muiltDense_x0/gamma"/device:CPU:0*
_output_shapes
: *
T0
?
	Assign_45Assign'BackupVariables/FCU_muiltDense_x0/gammaread_45"/device:CPU:0*
_output_shapes
: *:
_class0
.,loc:@BackupVariables/FCU_muiltDense_x0/gamma*
T0
a
read_46IdentityOutput_/dense/kernel"/device:CPU:0*
_output_shapes

: *
T0
?
	Assign_46Assign$BackupVariables/Output_/dense/kernelread_46"/device:CPU:0*
_output_shapes

: *
T0*7
_class-
+)loc:@BackupVariables/Output_/dense/kernel
[
read_47IdentityOutput_/dense/bias"/device:CPU:0*
_output_shapes
:*
T0
?
	Assign_47Assign"BackupVariables/Output_/dense/biasread_47"/device:CPU:0*5
_class+
)'loc:@BackupVariables/Output_/dense/bias*
_output_shapes
:*
T0
?
group_deps_2NoOp
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47"/device:CPU:0
}
read_48Identity+BackupVariables/CCN_1Conv_x0/convA10/kernel"/device:CPU:0*
T0*#
_output_shapes
:?
?
	Assign_48AssignCCN_1Conv_x0/convA10/kernelread_48"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?*
T0
s
read_49Identity)BackupVariables/CCN_1Conv_x0/convA10/bias"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_49AssignCCN_1Conv_x0/convA10/biasread_49"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?
~
read_50Identity+BackupVariables/CCN_1Conv_x0/convB10/kernel"/device:CPU:0*$
_output_shapes
:??*
T0
?
	Assign_50AssignCCN_1Conv_x0/convB10/kernelread_50"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0*$
_output_shapes
:??
s
read_51Identity)BackupVariables/CCN_1Conv_x0/convB10/bias"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_51AssignCCN_1Conv_x0/convB10/biasread_51"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0*
_output_shapes	
:?
~
read_52Identity+BackupVariables/CCN_1Conv_x0/convB20/kernel"/device:CPU:0*
T0*$
_output_shapes
:??
?
	Assign_52AssignCCN_1Conv_x0/convB20/kernelread_52"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0
s
read_53Identity)BackupVariables/CCN_1Conv_x0/convB20/bias"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_53AssignCCN_1Conv_x0/convB20/biasread_53"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
~
read_54Identity+BackupVariables/CCN_1Conv_x0/convA11/kernel"/device:CPU:0*
T0*$
_output_shapes
:??
?
	Assign_54AssignCCN_1Conv_x0/convA11/kernelread_54"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??*
T0
s
read_55Identity)BackupVariables/CCN_1Conv_x0/convA11/bias"/device:CPU:0*
T0*
_output_shapes	
:?
?
	Assign_55AssignCCN_1Conv_x0/convA11/biasread_55"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0*
_output_shapes	
:?
~
read_56Identity+BackupVariables/CCN_1Conv_x0/convB11/kernel"/device:CPU:0*
T0*$
_output_shapes
:??
?
	Assign_56AssignCCN_1Conv_x0/convB11/kernelread_56"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0*$
_output_shapes
:??
s
read_57Identity)BackupVariables/CCN_1Conv_x0/convB11/bias"/device:CPU:0*
T0*
_output_shapes	
:?
?
	Assign_57AssignCCN_1Conv_x0/convB11/biasread_57"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
~
read_58Identity+BackupVariables/CCN_1Conv_x0/convB21/kernel"/device:CPU:0*
T0*$
_output_shapes
:??
?
	Assign_58AssignCCN_1Conv_x0/convB21/kernelread_58"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
s
read_59Identity)BackupVariables/CCN_1Conv_x0/convB21/bias"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_59AssignCCN_1Conv_x0/convB21/biasread_59"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
i
read_60IdentityBackupVariables/Conv_out__/beta"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_60AssignConv_out__/betaread_60"/device:GPU:1*
T0*"
_class
loc:@Conv_out__/beta*
_output_shapes	
:?
j
read_61Identity BackupVariables/Conv_out__/gamma"/device:CPU:0*
_output_shapes	
:?*
T0
?
	Assign_61AssignConv_out__/gammaread_61"/device:GPU:1*
T0*#
_class
loc:@Conv_out__/gamma*
_output_shapes	
:?
?
read_62Identity2BackupVariables/Reconstruction_Output/dense/kernel"/device:CPU:0*
T0*
_output_shapes
:	?
?
	Assign_62Assign"Reconstruction_Output/dense/kernelread_62"/device:GPU:1*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?
y
read_63Identity0BackupVariables/Reconstruction_Output/dense/bias"/device:CPU:0*
T0*
_output_shapes
:
?
	Assign_63Assign Reconstruction_Output/dense/biasread_63"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
T0*
_output_shapes
:
j
read_64IdentityBackupVariables/dense/kernel"/device:CPU:0*
_output_shapes
:	? *
T0
?
	Assign_64Assigndense/kernelread_64"/device:GPU:1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	? 
c
read_65IdentityBackupVariables/dense/bias"/device:CPU:0*
_output_shapes
: *
T0
{
	Assign_65Assign
dense/biasread_65"/device:GPU:1*
_class
loc:@dense/bias*
_output_shapes
: *
T0
{
read_66Identity.BackupVariables/FCU_muiltDense_x0/dense/kernel"/device:CPU:0*
T0*
_output_shapes

:  
?
	Assign_66AssignFCU_muiltDense_x0/dense/kernelread_66"/device:GPU:1*
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes

:  
u
read_67Identity,BackupVariables/FCU_muiltDense_x0/dense/bias"/device:CPU:0*
T0*
_output_shapes
: 
?
	Assign_67AssignFCU_muiltDense_x0/dense/biasread_67"/device:GPU:1*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0
o
read_68Identity&BackupVariables/FCU_muiltDense_x0/beta"/device:CPU:0*
_output_shapes
: *
T0
?
	Assign_68AssignFCU_muiltDense_x0/betaread_68"/device:GPU:1*
T0*)
_class
loc:@FCU_muiltDense_x0/beta*
_output_shapes
: 
p
read_69Identity'BackupVariables/FCU_muiltDense_x0/gamma"/device:CPU:0*
_output_shapes
: *
T0
?
	Assign_69AssignFCU_muiltDense_x0/gammaread_69"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: *
T0
q
read_70Identity$BackupVariables/Output_/dense/kernel"/device:CPU:0*
_output_shapes

: *
T0
?
	Assign_70AssignOutput_/dense/kernelread_70"/device:GPU:1*
_output_shapes

: *
T0*'
_class
loc:@Output_/dense/kernel
k
read_71Identity"BackupVariables/Output_/dense/bias"/device:CPU:0*
_output_shapes
:*
T0
?
	Assign_71AssignOutput_/dense/biasread_71"/device:GPU:1*
T0*
_output_shapes
:*%
_class
loc:@Output_/dense/bias
?
group_deps_3NoOp
^Assign_48
^Assign_49
^Assign_50
^Assign_51
^Assign_52
^Assign_53
^Assign_54
^Assign_55
^Assign_56
^Assign_57
^Assign_58
^Assign_59
^Assign_60
^Assign_61
^Assign_62
^Assign_63
^Assign_64
^Assign_65
^Assign_66
^Assign_67
^Assign_68
^Assign_69
^Assign_70
^Assign_71"/device:GPU:1
h
save/filename/inputConst"/device:CPU:0*
_output_shapes
: *
valueB Bmodel*
dtype0
}
save/filenamePlaceholderWithDefaultsave/filename/input"/device:CPU:0*
shape: *
dtype0*
_output_shapes
: 
t

save/ConstPlaceholderWithDefaultsave/filename"/device:CPU:0*
_output_shapes
: *
dtype0*
shape: 
?#
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:{*?"
value?"B?"{B)BackupVariables/CCN_1Conv_x0/convA10/biasB+BackupVariables/CCN_1Conv_x0/convA10/kernelB)BackupVariables/CCN_1Conv_x0/convA11/biasB+BackupVariables/CCN_1Conv_x0/convA11/kernelB)BackupVariables/CCN_1Conv_x0/convB10/biasB+BackupVariables/CCN_1Conv_x0/convB10/kernelB)BackupVariables/CCN_1Conv_x0/convB11/biasB+BackupVariables/CCN_1Conv_x0/convB11/kernelB)BackupVariables/CCN_1Conv_x0/convB20/biasB+BackupVariables/CCN_1Conv_x0/convB20/kernelB)BackupVariables/CCN_1Conv_x0/convB21/biasB+BackupVariables/CCN_1Conv_x0/convB21/kernelBBackupVariables/Conv_out__/betaB BackupVariables/Conv_out__/gammaB&BackupVariables/FCU_muiltDense_x0/betaB,BackupVariables/FCU_muiltDense_x0/dense/biasB.BackupVariables/FCU_muiltDense_x0/dense/kernelB'BackupVariables/FCU_muiltDense_x0/gammaB"BackupVariables/Output_/dense/biasB$BackupVariables/Output_/dense/kernelB0BackupVariables/Reconstruction_Output/dense/biasB2BackupVariables/Reconstruction_Output/dense/kernelBBackupVariables/dense/biasBBackupVariables/dense/kernelBCCN_1Conv_x0/convA10/biasBCCN_1Conv_x0/convA10/bias/AdamB CCN_1Conv_x0/convA10/bias/Adam_1B2CCN_1Conv_x0/convA10/bias/ExponentialMovingAverageBCCN_1Conv_x0/convA10/kernelB CCN_1Conv_x0/convA10/kernel/AdamB"CCN_1Conv_x0/convA10/kernel/Adam_1B4CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convA11/biasBCCN_1Conv_x0/convA11/bias/AdamB CCN_1Conv_x0/convA11/bias/Adam_1B2CCN_1Conv_x0/convA11/bias/ExponentialMovingAverageBCCN_1Conv_x0/convA11/kernelB CCN_1Conv_x0/convA11/kernel/AdamB"CCN_1Conv_x0/convA11/kernel/Adam_1B4CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB10/biasBCCN_1Conv_x0/convB10/bias/AdamB CCN_1Conv_x0/convB10/bias/Adam_1B2CCN_1Conv_x0/convB10/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB10/kernelB CCN_1Conv_x0/convB10/kernel/AdamB"CCN_1Conv_x0/convB10/kernel/Adam_1B4CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB11/biasBCCN_1Conv_x0/convB11/bias/AdamB CCN_1Conv_x0/convB11/bias/Adam_1B2CCN_1Conv_x0/convB11/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB11/kernelB CCN_1Conv_x0/convB11/kernel/AdamB"CCN_1Conv_x0/convB11/kernel/Adam_1B4CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB20/biasBCCN_1Conv_x0/convB20/bias/AdamB CCN_1Conv_x0/convB20/bias/Adam_1B2CCN_1Conv_x0/convB20/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB20/kernelB CCN_1Conv_x0/convB20/kernel/AdamB"CCN_1Conv_x0/convB20/kernel/Adam_1B4CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB21/biasBCCN_1Conv_x0/convB21/bias/AdamB CCN_1Conv_x0/convB21/bias/Adam_1B2CCN_1Conv_x0/convB21/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB21/kernelB CCN_1Conv_x0/convB21/kernel/AdamB"CCN_1Conv_x0/convB21/kernel/Adam_1B4CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverageBConv_out__/betaBConv_out__/beta/AdamBConv_out__/beta/Adam_1B(Conv_out__/beta/ExponentialMovingAverageBConv_out__/gammaBConv_out__/gamma/AdamBConv_out__/gamma/Adam_1B)Conv_out__/gamma/ExponentialMovingAverageBFCU_muiltDense_x0/betaBFCU_muiltDense_x0/beta/AdamBFCU_muiltDense_x0/beta/Adam_1B/FCU_muiltDense_x0/beta/ExponentialMovingAverageBFCU_muiltDense_x0/dense/biasB!FCU_muiltDense_x0/dense/bias/AdamB#FCU_muiltDense_x0/dense/bias/Adam_1B5FCU_muiltDense_x0/dense/bias/ExponentialMovingAverageBFCU_muiltDense_x0/dense/kernelB#FCU_muiltDense_x0/dense/kernel/AdamB%FCU_muiltDense_x0/dense/kernel/Adam_1B7FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverageBFCU_muiltDense_x0/gammaBFCU_muiltDense_x0/gamma/AdamBFCU_muiltDense_x0/gamma/Adam_1B0FCU_muiltDense_x0/gamma/ExponentialMovingAverageBOutput_/dense/biasBOutput_/dense/bias/AdamBOutput_/dense/bias/Adam_1B+Output_/dense/bias/ExponentialMovingAverageBOutput_/dense/kernelBOutput_/dense/kernel/AdamBOutput_/dense/kernel/Adam_1B-Output_/dense/kernel/ExponentialMovingAverageB Reconstruction_Output/dense/biasB%Reconstruction_Output/dense/bias/AdamB'Reconstruction_Output/dense/bias/Adam_1B9Reconstruction_Output/dense/bias/ExponentialMovingAverageB"Reconstruction_Output/dense/kernelB'Reconstruction_Output/dense/kernel/AdamB)Reconstruction_Output/dense/kernel/Adam_1B;Reconstruction_Output/dense/kernel/ExponentialMovingAverageBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1B#dense/bias/ExponentialMovingAverageBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1B%dense/kernel/ExponentialMovingAverageBglobal_step*
dtype0
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:{*
dtype0*?
value?B?{B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?$
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices)BackupVariables/CCN_1Conv_x0/convA10/bias+BackupVariables/CCN_1Conv_x0/convA10/kernel)BackupVariables/CCN_1Conv_x0/convA11/bias+BackupVariables/CCN_1Conv_x0/convA11/kernel)BackupVariables/CCN_1Conv_x0/convB10/bias+BackupVariables/CCN_1Conv_x0/convB10/kernel)BackupVariables/CCN_1Conv_x0/convB11/bias+BackupVariables/CCN_1Conv_x0/convB11/kernel)BackupVariables/CCN_1Conv_x0/convB20/bias+BackupVariables/CCN_1Conv_x0/convB20/kernel)BackupVariables/CCN_1Conv_x0/convB21/bias+BackupVariables/CCN_1Conv_x0/convB21/kernelBackupVariables/Conv_out__/beta BackupVariables/Conv_out__/gamma&BackupVariables/FCU_muiltDense_x0/beta,BackupVariables/FCU_muiltDense_x0/dense/bias.BackupVariables/FCU_muiltDense_x0/dense/kernel'BackupVariables/FCU_muiltDense_x0/gamma"BackupVariables/Output_/dense/bias$BackupVariables/Output_/dense/kernel0BackupVariables/Reconstruction_Output/dense/bias2BackupVariables/Reconstruction_Output/dense/kernelBackupVariables/dense/biasBackupVariables/dense/kernelCCN_1Conv_x0/convA10/biasCCN_1Conv_x0/convA10/bias/Adam CCN_1Conv_x0/convA10/bias/Adam_12CCN_1Conv_x0/convA10/bias/ExponentialMovingAverageCCN_1Conv_x0/convA10/kernel CCN_1Conv_x0/convA10/kernel/Adam"CCN_1Conv_x0/convA10/kernel/Adam_14CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverageCCN_1Conv_x0/convA11/biasCCN_1Conv_x0/convA11/bias/Adam CCN_1Conv_x0/convA11/bias/Adam_12CCN_1Conv_x0/convA11/bias/ExponentialMovingAverageCCN_1Conv_x0/convA11/kernel CCN_1Conv_x0/convA11/kernel/Adam"CCN_1Conv_x0/convA11/kernel/Adam_14CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverageCCN_1Conv_x0/convB10/biasCCN_1Conv_x0/convB10/bias/Adam CCN_1Conv_x0/convB10/bias/Adam_12CCN_1Conv_x0/convB10/bias/ExponentialMovingAverageCCN_1Conv_x0/convB10/kernel CCN_1Conv_x0/convB10/kernel/Adam"CCN_1Conv_x0/convB10/kernel/Adam_14CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverageCCN_1Conv_x0/convB11/biasCCN_1Conv_x0/convB11/bias/Adam CCN_1Conv_x0/convB11/bias/Adam_12CCN_1Conv_x0/convB11/bias/ExponentialMovingAverageCCN_1Conv_x0/convB11/kernel CCN_1Conv_x0/convB11/kernel/Adam"CCN_1Conv_x0/convB11/kernel/Adam_14CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverageCCN_1Conv_x0/convB20/biasCCN_1Conv_x0/convB20/bias/Adam CCN_1Conv_x0/convB20/bias/Adam_12CCN_1Conv_x0/convB20/bias/ExponentialMovingAverageCCN_1Conv_x0/convB20/kernel CCN_1Conv_x0/convB20/kernel/Adam"CCN_1Conv_x0/convB20/kernel/Adam_14CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverageCCN_1Conv_x0/convB21/biasCCN_1Conv_x0/convB21/bias/Adam CCN_1Conv_x0/convB21/bias/Adam_12CCN_1Conv_x0/convB21/bias/ExponentialMovingAverageCCN_1Conv_x0/convB21/kernel CCN_1Conv_x0/convB21/kernel/Adam"CCN_1Conv_x0/convB21/kernel/Adam_14CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverageConv_out__/betaConv_out__/beta/AdamConv_out__/beta/Adam_1(Conv_out__/beta/ExponentialMovingAverageConv_out__/gammaConv_out__/gamma/AdamConv_out__/gamma/Adam_1)Conv_out__/gamma/ExponentialMovingAverageFCU_muiltDense_x0/betaFCU_muiltDense_x0/beta/AdamFCU_muiltDense_x0/beta/Adam_1/FCU_muiltDense_x0/beta/ExponentialMovingAverageFCU_muiltDense_x0/dense/bias!FCU_muiltDense_x0/dense/bias/Adam#FCU_muiltDense_x0/dense/bias/Adam_15FCU_muiltDense_x0/dense/bias/ExponentialMovingAverageFCU_muiltDense_x0/dense/kernel#FCU_muiltDense_x0/dense/kernel/Adam%FCU_muiltDense_x0/dense/kernel/Adam_17FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverageFCU_muiltDense_x0/gammaFCU_muiltDense_x0/gamma/AdamFCU_muiltDense_x0/gamma/Adam_10FCU_muiltDense_x0/gamma/ExponentialMovingAverageOutput_/dense/biasOutput_/dense/bias/AdamOutput_/dense/bias/Adam_1+Output_/dense/bias/ExponentialMovingAverageOutput_/dense/kernelOutput_/dense/kernel/AdamOutput_/dense/kernel/Adam_1-Output_/dense/kernel/ExponentialMovingAverage Reconstruction_Output/dense/bias%Reconstruction_Output/dense/bias/Adam'Reconstruction_Output/dense/bias/Adam_19Reconstruction_Output/dense/bias/ExponentialMovingAverage"Reconstruction_Output/dense/kernel'Reconstruction_Output/dense/kernel/Adam)Reconstruction_Output/dense/kernel/Adam_1;Reconstruction_Output/dense/kernel/ExponentialMovingAveragebeta1_powerbeta2_power
dense/biasdense/bias/Adamdense/bias/Adam_1#dense/bias/ExponentialMovingAveragedense/kerneldense/kernel/Adamdense/kernel/Adam_1%dense/kernel/ExponentialMovingAverageglobal_step"/device:CPU:0*?
dtypes
}2{
?
save/control_dependencyIdentity
save/Const^save/SaveV2"/device:CPU:0*
T0*
_class
loc:@save/Const*
_output_shapes
: 
?#
save/RestoreV2/tensor_namesConst"/device:CPU:0*?"
value?"B?"{B)BackupVariables/CCN_1Conv_x0/convA10/biasB+BackupVariables/CCN_1Conv_x0/convA10/kernelB)BackupVariables/CCN_1Conv_x0/convA11/biasB+BackupVariables/CCN_1Conv_x0/convA11/kernelB)BackupVariables/CCN_1Conv_x0/convB10/biasB+BackupVariables/CCN_1Conv_x0/convB10/kernelB)BackupVariables/CCN_1Conv_x0/convB11/biasB+BackupVariables/CCN_1Conv_x0/convB11/kernelB)BackupVariables/CCN_1Conv_x0/convB20/biasB+BackupVariables/CCN_1Conv_x0/convB20/kernelB)BackupVariables/CCN_1Conv_x0/convB21/biasB+BackupVariables/CCN_1Conv_x0/convB21/kernelBBackupVariables/Conv_out__/betaB BackupVariables/Conv_out__/gammaB&BackupVariables/FCU_muiltDense_x0/betaB,BackupVariables/FCU_muiltDense_x0/dense/biasB.BackupVariables/FCU_muiltDense_x0/dense/kernelB'BackupVariables/FCU_muiltDense_x0/gammaB"BackupVariables/Output_/dense/biasB$BackupVariables/Output_/dense/kernelB0BackupVariables/Reconstruction_Output/dense/biasB2BackupVariables/Reconstruction_Output/dense/kernelBBackupVariables/dense/biasBBackupVariables/dense/kernelBCCN_1Conv_x0/convA10/biasBCCN_1Conv_x0/convA10/bias/AdamB CCN_1Conv_x0/convA10/bias/Adam_1B2CCN_1Conv_x0/convA10/bias/ExponentialMovingAverageBCCN_1Conv_x0/convA10/kernelB CCN_1Conv_x0/convA10/kernel/AdamB"CCN_1Conv_x0/convA10/kernel/Adam_1B4CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convA11/biasBCCN_1Conv_x0/convA11/bias/AdamB CCN_1Conv_x0/convA11/bias/Adam_1B2CCN_1Conv_x0/convA11/bias/ExponentialMovingAverageBCCN_1Conv_x0/convA11/kernelB CCN_1Conv_x0/convA11/kernel/AdamB"CCN_1Conv_x0/convA11/kernel/Adam_1B4CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB10/biasBCCN_1Conv_x0/convB10/bias/AdamB CCN_1Conv_x0/convB10/bias/Adam_1B2CCN_1Conv_x0/convB10/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB10/kernelB CCN_1Conv_x0/convB10/kernel/AdamB"CCN_1Conv_x0/convB10/kernel/Adam_1B4CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB11/biasBCCN_1Conv_x0/convB11/bias/AdamB CCN_1Conv_x0/convB11/bias/Adam_1B2CCN_1Conv_x0/convB11/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB11/kernelB CCN_1Conv_x0/convB11/kernel/AdamB"CCN_1Conv_x0/convB11/kernel/Adam_1B4CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB20/biasBCCN_1Conv_x0/convB20/bias/AdamB CCN_1Conv_x0/convB20/bias/Adam_1B2CCN_1Conv_x0/convB20/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB20/kernelB CCN_1Conv_x0/convB20/kernel/AdamB"CCN_1Conv_x0/convB20/kernel/Adam_1B4CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB21/biasBCCN_1Conv_x0/convB21/bias/AdamB CCN_1Conv_x0/convB21/bias/Adam_1B2CCN_1Conv_x0/convB21/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB21/kernelB CCN_1Conv_x0/convB21/kernel/AdamB"CCN_1Conv_x0/convB21/kernel/Adam_1B4CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverageBConv_out__/betaBConv_out__/beta/AdamBConv_out__/beta/Adam_1B(Conv_out__/beta/ExponentialMovingAverageBConv_out__/gammaBConv_out__/gamma/AdamBConv_out__/gamma/Adam_1B)Conv_out__/gamma/ExponentialMovingAverageBFCU_muiltDense_x0/betaBFCU_muiltDense_x0/beta/AdamBFCU_muiltDense_x0/beta/Adam_1B/FCU_muiltDense_x0/beta/ExponentialMovingAverageBFCU_muiltDense_x0/dense/biasB!FCU_muiltDense_x0/dense/bias/AdamB#FCU_muiltDense_x0/dense/bias/Adam_1B5FCU_muiltDense_x0/dense/bias/ExponentialMovingAverageBFCU_muiltDense_x0/dense/kernelB#FCU_muiltDense_x0/dense/kernel/AdamB%FCU_muiltDense_x0/dense/kernel/Adam_1B7FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverageBFCU_muiltDense_x0/gammaBFCU_muiltDense_x0/gamma/AdamBFCU_muiltDense_x0/gamma/Adam_1B0FCU_muiltDense_x0/gamma/ExponentialMovingAverageBOutput_/dense/biasBOutput_/dense/bias/AdamBOutput_/dense/bias/Adam_1B+Output_/dense/bias/ExponentialMovingAverageBOutput_/dense/kernelBOutput_/dense/kernel/AdamBOutput_/dense/kernel/Adam_1B-Output_/dense/kernel/ExponentialMovingAverageB Reconstruction_Output/dense/biasB%Reconstruction_Output/dense/bias/AdamB'Reconstruction_Output/dense/bias/Adam_1B9Reconstruction_Output/dense/bias/ExponentialMovingAverageB"Reconstruction_Output/dense/kernelB'Reconstruction_Output/dense/kernel/AdamB)Reconstruction_Output/dense/kernel/Adam_1B;Reconstruction_Output/dense/kernel/ExponentialMovingAverageBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1B#dense/bias/ExponentialMovingAverageBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1B%dense/kernel/ExponentialMovingAverageBglobal_step*
dtype0*
_output_shapes
:{
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*?
value?B?{B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:{*
dtype0
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*?
dtypes
}2{*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save/AssignAssign)BackupVariables/CCN_1Conv_x0/convA10/biassave/RestoreV2"/device:CPU:0*
T0*
_output_shapes	
:?*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convA10/bias
?
save/Assign_1Assign+BackupVariables/CCN_1Conv_x0/convA10/kernelsave/RestoreV2:1"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?*
T0
?
save/Assign_2Assign)BackupVariables/CCN_1Conv_x0/convA11/biassave/RestoreV2:2"/device:CPU:0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convA11/bias*
T0*
_output_shapes	
:?
?
save/Assign_3Assign+BackupVariables/CCN_1Conv_x0/convA11/kernelsave/RestoreV2:3"/device:CPU:0*$
_output_shapes
:??*
T0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convA11/kernel
?
save/Assign_4Assign)BackupVariables/CCN_1Conv_x0/convB10/biassave/RestoreV2:4"/device:CPU:0*
_output_shapes	
:?*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB10/bias*
T0
?
save/Assign_5Assign+BackupVariables/CCN_1Conv_x0/convB10/kernelsave/RestoreV2:5"/device:CPU:0*$
_output_shapes
:??*
T0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB10/kernel
?
save/Assign_6Assign)BackupVariables/CCN_1Conv_x0/convB11/biassave/RestoreV2:6"/device:CPU:0*
_output_shapes	
:?*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB11/bias
?
save/Assign_7Assign+BackupVariables/CCN_1Conv_x0/convB11/kernelsave/RestoreV2:7"/device:CPU:0*
T0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB11/kernel*$
_output_shapes
:??
?
save/Assign_8Assign)BackupVariables/CCN_1Conv_x0/convB20/biassave/RestoreV2:8"/device:CPU:0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB20/bias*
_output_shapes	
:?*
T0
?
save/Assign_9Assign+BackupVariables/CCN_1Conv_x0/convB20/kernelsave/RestoreV2:9"/device:CPU:0*
T0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??
?
save/Assign_10Assign)BackupVariables/CCN_1Conv_x0/convB21/biassave/RestoreV2:10"/device:CPU:0*
_output_shapes	
:?*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB21/bias*
T0
?
save/Assign_11Assign+BackupVariables/CCN_1Conv_x0/convB21/kernelsave/RestoreV2:11"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB21/kernel*
T0*$
_output_shapes
:??
?
save/Assign_12AssignBackupVariables/Conv_out__/betasave/RestoreV2:12"/device:CPU:0*
T0*
_output_shapes	
:?*2
_class(
&$loc:@BackupVariables/Conv_out__/beta
?
save/Assign_13Assign BackupVariables/Conv_out__/gammasave/RestoreV2:13"/device:CPU:0*
T0*3
_class)
'%loc:@BackupVariables/Conv_out__/gamma*
_output_shapes	
:?
?
save/Assign_14Assign&BackupVariables/FCU_muiltDense_x0/betasave/RestoreV2:14"/device:CPU:0*
_output_shapes
: *
T0*9
_class/
-+loc:@BackupVariables/FCU_muiltDense_x0/beta
?
save/Assign_15Assign,BackupVariables/FCU_muiltDense_x0/dense/biassave/RestoreV2:15"/device:CPU:0*?
_class5
31loc:@BackupVariables/FCU_muiltDense_x0/dense/bias*
_output_shapes
: *
T0
?
save/Assign_16Assign.BackupVariables/FCU_muiltDense_x0/dense/kernelsave/RestoreV2:16"/device:CPU:0*
_output_shapes

:  *
T0*A
_class7
53loc:@BackupVariables/FCU_muiltDense_x0/dense/kernel
?
save/Assign_17Assign'BackupVariables/FCU_muiltDense_x0/gammasave/RestoreV2:17"/device:CPU:0*:
_class0
.,loc:@BackupVariables/FCU_muiltDense_x0/gamma*
_output_shapes
: *
T0
?
save/Assign_18Assign"BackupVariables/Output_/dense/biassave/RestoreV2:18"/device:CPU:0*
_output_shapes
:*5
_class+
)'loc:@BackupVariables/Output_/dense/bias*
T0
?
save/Assign_19Assign$BackupVariables/Output_/dense/kernelsave/RestoreV2:19"/device:CPU:0*
T0*
_output_shapes

: *7
_class-
+)loc:@BackupVariables/Output_/dense/kernel
?
save/Assign_20Assign0BackupVariables/Reconstruction_Output/dense/biassave/RestoreV2:20"/device:CPU:0*
_output_shapes
:*C
_class9
75loc:@BackupVariables/Reconstruction_Output/dense/bias*
T0
?
save/Assign_21Assign2BackupVariables/Reconstruction_Output/dense/kernelsave/RestoreV2:21"/device:CPU:0*
_output_shapes
:	?*
T0*E
_class;
97loc:@BackupVariables/Reconstruction_Output/dense/kernel
?
save/Assign_22AssignBackupVariables/dense/biassave/RestoreV2:22"/device:CPU:0*
T0*
_output_shapes
: *-
_class#
!loc:@BackupVariables/dense/bias
?
save/Assign_23AssignBackupVariables/dense/kernelsave/RestoreV2:23"/device:CPU:0*
_output_shapes
:	? *
T0*/
_class%
#!loc:@BackupVariables/dense/kernel
?
save/Assign_24AssignCCN_1Conv_x0/convA10/biassave/RestoreV2:24"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
save/Assign_25AssignCCN_1Conv_x0/convA10/bias/Adamsave/RestoreV2:25"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0*
_output_shapes	
:?
?
save/Assign_26Assign CCN_1Conv_x0/convA10/bias/Adam_1save/RestoreV2:26"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?*
T0
?
save/Assign_27Assign2CCN_1Conv_x0/convA10/bias/ExponentialMovingAveragesave/RestoreV2:27"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
save/Assign_28AssignCCN_1Conv_x0/convA10/kernelsave/RestoreV2:28"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0*#
_output_shapes
:?
?
save/Assign_29Assign CCN_1Conv_x0/convA10/kernel/Adamsave/RestoreV2:29"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0*#
_output_shapes
:?
?
save/Assign_30Assign"CCN_1Conv_x0/convA10/kernel/Adam_1save/RestoreV2:30"/device:GPU:1*#
_output_shapes
:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0
?
save/Assign_31Assign4CCN_1Conv_x0/convA10/kernel/ExponentialMovingAveragesave/RestoreV2:31"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?
?
save/Assign_32AssignCCN_1Conv_x0/convA11/biassave/RestoreV2:32"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0*
_output_shapes	
:?
?
save/Assign_33AssignCCN_1Conv_x0/convA11/bias/Adamsave/RestoreV2:33"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0
?
save/Assign_34Assign CCN_1Conv_x0/convA11/bias/Adam_1save/RestoreV2:34"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?*
T0
?
save/Assign_35Assign2CCN_1Conv_x0/convA11/bias/ExponentialMovingAveragesave/RestoreV2:35"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
_output_shapes	
:?
?
save/Assign_36AssignCCN_1Conv_x0/convA11/kernelsave/RestoreV2:36"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
save/Assign_37Assign CCN_1Conv_x0/convA11/kernel/Adamsave/RestoreV2:37"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??*
T0
?
save/Assign_38Assign"CCN_1Conv_x0/convA11/kernel/Adam_1save/RestoreV2:38"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0
?
save/Assign_39Assign4CCN_1Conv_x0/convA11/kernel/ExponentialMovingAveragesave/RestoreV2:39"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??
?
save/Assign_40AssignCCN_1Conv_x0/convB10/biassave/RestoreV2:40"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0
?
save/Assign_41AssignCCN_1Conv_x0/convB10/bias/Adamsave/RestoreV2:41"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
?
save/Assign_42Assign CCN_1Conv_x0/convB10/bias/Adam_1save/RestoreV2:42"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0*
_output_shapes	
:?
?
save/Assign_43Assign2CCN_1Conv_x0/convB10/bias/ExponentialMovingAveragesave/RestoreV2:43"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
_output_shapes	
:?*
T0
?
save/Assign_44AssignCCN_1Conv_x0/convB10/kernelsave/RestoreV2:44"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0
?
save/Assign_45Assign CCN_1Conv_x0/convB10/kernel/Adamsave/RestoreV2:45"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel
?
save/Assign_46Assign"CCN_1Conv_x0/convB10/kernel/Adam_1save/RestoreV2:46"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??
?
save/Assign_47Assign4CCN_1Conv_x0/convB10/kernel/ExponentialMovingAveragesave/RestoreV2:47"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??
?
save/Assign_48AssignCCN_1Conv_x0/convB11/biassave/RestoreV2:48"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
T0*
_output_shapes	
:?
?
save/Assign_49AssignCCN_1Conv_x0/convB11/bias/Adamsave/RestoreV2:49"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
save/Assign_50Assign CCN_1Conv_x0/convB11/bias/Adam_1save/RestoreV2:50"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
T0*
_output_shapes	
:?
?
save/Assign_51Assign2CCN_1Conv_x0/convB11/bias/ExponentialMovingAveragesave/RestoreV2:51"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias
?
save/Assign_52AssignCCN_1Conv_x0/convB11/kernelsave/RestoreV2:52"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*$
_output_shapes
:??*
T0
?
save/Assign_53Assign CCN_1Conv_x0/convB11/kernel/Adamsave/RestoreV2:53"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*$
_output_shapes
:??*
T0
?
save/Assign_54Assign"CCN_1Conv_x0/convB11/kernel/Adam_1save/RestoreV2:54"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0*$
_output_shapes
:??
?
save/Assign_55Assign4CCN_1Conv_x0/convB11/kernel/ExponentialMovingAveragesave/RestoreV2:55"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
save/Assign_56AssignCCN_1Conv_x0/convB20/biassave/RestoreV2:56"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
_output_shapes	
:?
?
save/Assign_57AssignCCN_1Conv_x0/convB20/bias/Adamsave/RestoreV2:57"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
save/Assign_58Assign CCN_1Conv_x0/convB20/bias/Adam_1save/RestoreV2:58"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
_output_shapes	
:?
?
save/Assign_59Assign2CCN_1Conv_x0/convB20/bias/ExponentialMovingAveragesave/RestoreV2:59"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
T0*
_output_shapes	
:?
?
save/Assign_60AssignCCN_1Conv_x0/convB20/kernelsave/RestoreV2:60"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??*
T0
?
save/Assign_61Assign CCN_1Conv_x0/convB20/kernel/Adamsave/RestoreV2:61"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??
?
save/Assign_62Assign"CCN_1Conv_x0/convB20/kernel/Adam_1save/RestoreV2:62"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??*
T0
?
save/Assign_63Assign4CCN_1Conv_x0/convB20/kernel/ExponentialMovingAveragesave/RestoreV2:63"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0*$
_output_shapes
:??
?
save/Assign_64AssignCCN_1Conv_x0/convB21/biassave/RestoreV2:64"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
save/Assign_65AssignCCN_1Conv_x0/convB21/bias/Adamsave/RestoreV2:65"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes	
:?*
T0
?
save/Assign_66Assign CCN_1Conv_x0/convB21/bias/Adam_1save/RestoreV2:66"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0
?
save/Assign_67Assign2CCN_1Conv_x0/convB21/bias/ExponentialMovingAveragesave/RestoreV2:67"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0*
_output_shapes	
:?
?
save/Assign_68AssignCCN_1Conv_x0/convB21/kernelsave/RestoreV2:68"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
T0*$
_output_shapes
:??
?
save/Assign_69Assign CCN_1Conv_x0/convB21/kernel/Adamsave/RestoreV2:69"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
save/Assign_70Assign"CCN_1Conv_x0/convB21/kernel/Adam_1save/RestoreV2:70"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
T0*$
_output_shapes
:??
?
save/Assign_71Assign4CCN_1Conv_x0/convB21/kernel/ExponentialMovingAveragesave/RestoreV2:71"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
save/Assign_72AssignConv_out__/betasave/RestoreV2:72"/device:GPU:1*
_output_shapes	
:?*"
_class
loc:@Conv_out__/beta*
T0
?
save/Assign_73AssignConv_out__/beta/Adamsave/RestoreV2:73"/device:GPU:1*"
_class
loc:@Conv_out__/beta*
T0*
_output_shapes	
:?
?
save/Assign_74AssignConv_out__/beta/Adam_1save/RestoreV2:74"/device:GPU:1*
T0*"
_class
loc:@Conv_out__/beta*
_output_shapes	
:?
?
save/Assign_75Assign(Conv_out__/beta/ExponentialMovingAveragesave/RestoreV2:75"/device:GPU:1*
T0*"
_class
loc:@Conv_out__/beta*
_output_shapes	
:?
?
save/Assign_76AssignConv_out__/gammasave/RestoreV2:76"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
T0*
_output_shapes	
:?
?
save/Assign_77AssignConv_out__/gamma/Adamsave/RestoreV2:77"/device:GPU:1*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma*
T0
?
save/Assign_78AssignConv_out__/gamma/Adam_1save/RestoreV2:78"/device:GPU:1*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma*
T0
?
save/Assign_79Assign)Conv_out__/gamma/ExponentialMovingAveragesave/RestoreV2:79"/device:GPU:1*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma*
T0
?
save/Assign_80AssignFCU_muiltDense_x0/betasave/RestoreV2:80"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta*
_output_shapes
: *
T0
?
save/Assign_81AssignFCU_muiltDense_x0/beta/Adamsave/RestoreV2:81"/device:GPU:1*
T0*)
_class
loc:@FCU_muiltDense_x0/beta*
_output_shapes
: 
?
save/Assign_82AssignFCU_muiltDense_x0/beta/Adam_1save/RestoreV2:82"/device:GPU:1*
_output_shapes
: *
T0*)
_class
loc:@FCU_muiltDense_x0/beta
?
save/Assign_83Assign/FCU_muiltDense_x0/beta/ExponentialMovingAveragesave/RestoreV2:83"/device:GPU:1*
_output_shapes
: *)
_class
loc:@FCU_muiltDense_x0/beta*
T0
?
save/Assign_84AssignFCU_muiltDense_x0/dense/biassave/RestoreV2:84"/device:GPU:1*
T0*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
save/Assign_85Assign!FCU_muiltDense_x0/dense/bias/Adamsave/RestoreV2:85"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
_output_shapes
: *
T0
?
save/Assign_86Assign#FCU_muiltDense_x0/dense/bias/Adam_1save/RestoreV2:86"/device:GPU:1*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0
?
save/Assign_87Assign5FCU_muiltDense_x0/dense/bias/ExponentialMovingAveragesave/RestoreV2:87"/device:GPU:1*
T0*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
save/Assign_88AssignFCU_muiltDense_x0/dense/kernelsave/RestoreV2:88"/device:GPU:1*
_output_shapes

:  *
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
save/Assign_89Assign#FCU_muiltDense_x0/dense/kernel/Adamsave/RestoreV2:89"/device:GPU:1*
T0*
_output_shapes

:  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
save/Assign_90Assign%FCU_muiltDense_x0/dense/kernel/Adam_1save/RestoreV2:90"/device:GPU:1*
_output_shapes

:  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
T0
?
save/Assign_91Assign7FCU_muiltDense_x0/dense/kernel/ExponentialMovingAveragesave/RestoreV2:91"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes

:  *
T0
?
save/Assign_92AssignFCU_muiltDense_x0/gammasave/RestoreV2:92"/device:GPU:1*
T0**
_class 
loc:@FCU_muiltDense_x0/gamma*
_output_shapes
: 
?
save/Assign_93AssignFCU_muiltDense_x0/gamma/Adamsave/RestoreV2:93"/device:GPU:1*
_output_shapes
: **
_class 
loc:@FCU_muiltDense_x0/gamma*
T0
?
save/Assign_94AssignFCU_muiltDense_x0/gamma/Adam_1save/RestoreV2:94"/device:GPU:1*
_output_shapes
: **
_class 
loc:@FCU_muiltDense_x0/gamma*
T0
?
save/Assign_95Assign0FCU_muiltDense_x0/gamma/ExponentialMovingAveragesave/RestoreV2:95"/device:GPU:1*
T0*
_output_shapes
: **
_class 
loc:@FCU_muiltDense_x0/gamma
?
save/Assign_96AssignOutput_/dense/biassave/RestoreV2:96"/device:GPU:1*
T0*
_output_shapes
:*%
_class
loc:@Output_/dense/bias
?
save/Assign_97AssignOutput_/dense/bias/Adamsave/RestoreV2:97"/device:GPU:1*
_output_shapes
:*
T0*%
_class
loc:@Output_/dense/bias
?
save/Assign_98AssignOutput_/dense/bias/Adam_1save/RestoreV2:98"/device:GPU:1*
T0*%
_class
loc:@Output_/dense/bias*
_output_shapes
:
?
save/Assign_99Assign+Output_/dense/bias/ExponentialMovingAveragesave/RestoreV2:99"/device:GPU:1*
T0*%
_class
loc:@Output_/dense/bias*
_output_shapes
:
?
save/Assign_100AssignOutput_/dense/kernelsave/RestoreV2:100"/device:GPU:1*
_output_shapes

: *
T0*'
_class
loc:@Output_/dense/kernel
?
save/Assign_101AssignOutput_/dense/kernel/Adamsave/RestoreV2:101"/device:GPU:1*
_output_shapes

: *
T0*'
_class
loc:@Output_/dense/kernel
?
save/Assign_102AssignOutput_/dense/kernel/Adam_1save/RestoreV2:102"/device:GPU:1*
_output_shapes

: *'
_class
loc:@Output_/dense/kernel*
T0
?
save/Assign_103Assign-Output_/dense/kernel/ExponentialMovingAveragesave/RestoreV2:103"/device:GPU:1*
_output_shapes

: *'
_class
loc:@Output_/dense/kernel*
T0
?
save/Assign_104Assign Reconstruction_Output/dense/biassave/RestoreV2:104"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
T0*
_output_shapes
:
?
save/Assign_105Assign%Reconstruction_Output/dense/bias/Adamsave/RestoreV2:105"/device:GPU:1*
T0*
_output_shapes
:*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
save/Assign_106Assign'Reconstruction_Output/dense/bias/Adam_1save/RestoreV2:106"/device:GPU:1*
_output_shapes
:*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
save/Assign_107Assign9Reconstruction_Output/dense/bias/ExponentialMovingAveragesave/RestoreV2:107"/device:GPU:1*
_output_shapes
:*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
save/Assign_108Assign"Reconstruction_Output/dense/kernelsave/RestoreV2:108"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?*
T0
?
save/Assign_109Assign'Reconstruction_Output/dense/kernel/Adamsave/RestoreV2:109"/device:GPU:1*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?
?
save/Assign_110Assign)Reconstruction_Output/dense/kernel/Adam_1save/RestoreV2:110"/device:GPU:1*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?
?
save/Assign_111Assign;Reconstruction_Output/dense/kernel/ExponentialMovingAveragesave/RestoreV2:111"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?*
T0
?
save/Assign_112Assignbeta1_powersave/RestoreV2:112"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes
: *
T0
?
save/Assign_113Assignbeta2_powersave/RestoreV2:113"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0*
_output_shapes
: 
?
save/Assign_114Assign
dense/biassave/RestoreV2:114"/device:GPU:1*
T0*
_output_shapes
: *
_class
loc:@dense/bias
?
save/Assign_115Assigndense/bias/Adamsave/RestoreV2:115"/device:GPU:1*
_class
loc:@dense/bias*
T0*
_output_shapes
: 
?
save/Assign_116Assigndense/bias/Adam_1save/RestoreV2:116"/device:GPU:1*
T0*
_output_shapes
: *
_class
loc:@dense/bias
?
save/Assign_117Assign#dense/bias/ExponentialMovingAveragesave/RestoreV2:117"/device:GPU:1*
_class
loc:@dense/bias*
_output_shapes
: *
T0
?
save/Assign_118Assigndense/kernelsave/RestoreV2:118"/device:GPU:1*
_class
loc:@dense/kernel*
_output_shapes
:	? *
T0
?
save/Assign_119Assigndense/kernel/Adamsave/RestoreV2:119"/device:GPU:1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	? 
?
save/Assign_120Assigndense/kernel/Adam_1save/RestoreV2:120"/device:GPU:1*
_class
loc:@dense/kernel*
T0*
_output_shapes
:	? 
?
save/Assign_121Assign%dense/kernel/ExponentialMovingAveragesave/RestoreV2:121"/device:GPU:1*
_output_shapes
:	? *
_class
loc:@dense/kernel*
T0
?
save/Assign_122Assignglobal_stepsave/RestoreV2:122"/device:CPU:0*
_class
loc:@global_step*
_output_shapes
: *
T0
?
save/restore_all/NoOpNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_122^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"/device:CPU:0
?
save/restore_all/NoOp_1NoOp^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_108^save/Assign_109^save/Assign_110^save/Assign_111^save/Assign_112^save/Assign_113^save/Assign_114^save/Assign_115^save/Assign_116^save/Assign_117^save/Assign_118^save/Assign_119^save/Assign_120^save/Assign_121^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99"/device:GPU:1
Y
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"/device:CPU:0
L
m_loss_Placeholder"/device:CPU:0*
_output_shapes
:*
dtype0
K
R_LossPlaceholder"/device:CPU:0*
_output_shapes
:*
dtype0
N
	Cust_LossPlaceholder"/device:CPU:0*
dtype0*
_output_shapes
:
P
custom_Acc_Placeholder"/device:CPU:0*
_output_shapes
:*
dtype0
H
figPlaceholder"/device:CPU:0*
dtype0*
_output_shapes
:
w
Result_items/Loss/tagsConst"/device:CPU:0*"
valueB BResult_items/Loss*
dtype0*
_output_shapes
: 
s
Result_items/LossScalarSummaryResult_items/Loss/tagsm_loss_"/device:CPU:0*
T0*
_output_shapes
: 

Result_items/Accuracy/tagsConst"/device:CPU:0*&
valueB BResult_items/Accuracy*
dtype0*
_output_shapes
: 

Result_items/AccuracyScalarSummaryResult_items/Accuracy/tagscustom_Acc_"/device:CPU:0*
_output_shapes
: *
T0
?
%Result_items/Reconstruction_Loss/tagsConst"/device:CPU:0*
dtype0*
_output_shapes
: *1
value(B& B Result_items/Reconstruction_Loss
?
 Result_items/Reconstruction_LossScalarSummary%Result_items/Reconstruction_Loss/tagsR_Loss"/device:CPU:0*
_output_shapes
: *
T0
?
Result_items/Custom_Loss/tagsConst"/device:CPU:0*)
value B BResult_items/Custom_Loss*
dtype0*
_output_shapes
: 
?
Result_items/Custom_LossScalarSummaryResult_items/Custom_Loss/tags	Cust_Loss"/device:CPU:0*
T0*
_output_shapes
: 
?
Merge/MergeSummaryMergeSummaryResult_items/LossResult_items/Accuracy Result_items/Reconstruction_LossResult_items/Custom_Loss"/device:CPU:0*
N*
_output_shapes
: 
?	
	init/NoOpNoOp1^BackupVariables/CCN_1Conv_x0/convA10/bias/Assign3^BackupVariables/CCN_1Conv_x0/convA10/kernel/Assign1^BackupVariables/CCN_1Conv_x0/convA11/bias/Assign3^BackupVariables/CCN_1Conv_x0/convA11/kernel/Assign1^BackupVariables/CCN_1Conv_x0/convB10/bias/Assign3^BackupVariables/CCN_1Conv_x0/convB10/kernel/Assign1^BackupVariables/CCN_1Conv_x0/convB11/bias/Assign3^BackupVariables/CCN_1Conv_x0/convB11/kernel/Assign1^BackupVariables/CCN_1Conv_x0/convB20/bias/Assign3^BackupVariables/CCN_1Conv_x0/convB20/kernel/Assign1^BackupVariables/CCN_1Conv_x0/convB21/bias/Assign3^BackupVariables/CCN_1Conv_x0/convB21/kernel/Assign'^BackupVariables/Conv_out__/beta/Assign(^BackupVariables/Conv_out__/gamma/Assign.^BackupVariables/FCU_muiltDense_x0/beta/Assign4^BackupVariables/FCU_muiltDense_x0/dense/bias/Assign6^BackupVariables/FCU_muiltDense_x0/dense/kernel/Assign/^BackupVariables/FCU_muiltDense_x0/gamma/Assign*^BackupVariables/Output_/dense/bias/Assign,^BackupVariables/Output_/dense/kernel/Assign8^BackupVariables/Reconstruction_Output/dense/bias/Assign:^BackupVariables/Reconstruction_Output/dense/kernel/Assign"^BackupVariables/dense/bias/Assign$^BackupVariables/dense/kernel/Assign^global_step/Assign"/device:CPU:0
? 
init/NoOp_1NoOp&^CCN_1Conv_x0/convA10/bias/Adam/Assign(^CCN_1Conv_x0/convA10/bias/Adam_1/Assign!^CCN_1Conv_x0/convA10/bias/Assign:^CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/Assign(^CCN_1Conv_x0/convA10/kernel/Adam/Assign*^CCN_1Conv_x0/convA10/kernel/Adam_1/Assign#^CCN_1Conv_x0/convA10/kernel/Assign<^CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/Assign&^CCN_1Conv_x0/convA11/bias/Adam/Assign(^CCN_1Conv_x0/convA11/bias/Adam_1/Assign!^CCN_1Conv_x0/convA11/bias/Assign:^CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/Assign(^CCN_1Conv_x0/convA11/kernel/Adam/Assign*^CCN_1Conv_x0/convA11/kernel/Adam_1/Assign#^CCN_1Conv_x0/convA11/kernel/Assign<^CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/Assign&^CCN_1Conv_x0/convB10/bias/Adam/Assign(^CCN_1Conv_x0/convB10/bias/Adam_1/Assign!^CCN_1Conv_x0/convB10/bias/Assign:^CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/Assign(^CCN_1Conv_x0/convB10/kernel/Adam/Assign*^CCN_1Conv_x0/convB10/kernel/Adam_1/Assign#^CCN_1Conv_x0/convB10/kernel/Assign<^CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/Assign&^CCN_1Conv_x0/convB11/bias/Adam/Assign(^CCN_1Conv_x0/convB11/bias/Adam_1/Assign!^CCN_1Conv_x0/convB11/bias/Assign:^CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/Assign(^CCN_1Conv_x0/convB11/kernel/Adam/Assign*^CCN_1Conv_x0/convB11/kernel/Adam_1/Assign#^CCN_1Conv_x0/convB11/kernel/Assign<^CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/Assign&^CCN_1Conv_x0/convB20/bias/Adam/Assign(^CCN_1Conv_x0/convB20/bias/Adam_1/Assign!^CCN_1Conv_x0/convB20/bias/Assign:^CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/Assign(^CCN_1Conv_x0/convB20/kernel/Adam/Assign*^CCN_1Conv_x0/convB20/kernel/Adam_1/Assign#^CCN_1Conv_x0/convB20/kernel/Assign<^CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/Assign&^CCN_1Conv_x0/convB21/bias/Adam/Assign(^CCN_1Conv_x0/convB21/bias/Adam_1/Assign!^CCN_1Conv_x0/convB21/bias/Assign:^CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/Assign(^CCN_1Conv_x0/convB21/kernel/Adam/Assign*^CCN_1Conv_x0/convB21/kernel/Adam_1/Assign#^CCN_1Conv_x0/convB21/kernel/Assign<^CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/Assign^Conv_out__/beta/Adam/Assign^Conv_out__/beta/Adam_1/Assign^Conv_out__/beta/Assign0^Conv_out__/beta/ExponentialMovingAverage/Assign^Conv_out__/gamma/Adam/Assign^Conv_out__/gamma/Adam_1/Assign^Conv_out__/gamma/Assign1^Conv_out__/gamma/ExponentialMovingAverage/Assign#^FCU_muiltDense_x0/beta/Adam/Assign%^FCU_muiltDense_x0/beta/Adam_1/Assign^FCU_muiltDense_x0/beta/Assign7^FCU_muiltDense_x0/beta/ExponentialMovingAverage/Assign)^FCU_muiltDense_x0/dense/bias/Adam/Assign+^FCU_muiltDense_x0/dense/bias/Adam_1/Assign$^FCU_muiltDense_x0/dense/bias/Assign=^FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/Assign+^FCU_muiltDense_x0/dense/kernel/Adam/Assign-^FCU_muiltDense_x0/dense/kernel/Adam_1/Assign&^FCU_muiltDense_x0/dense/kernel/Assign?^FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/Assign$^FCU_muiltDense_x0/gamma/Adam/Assign&^FCU_muiltDense_x0/gamma/Adam_1/Assign^FCU_muiltDense_x0/gamma/Assign8^FCU_muiltDense_x0/gamma/ExponentialMovingAverage/Assign^Output_/dense/bias/Adam/Assign!^Output_/dense/bias/Adam_1/Assign^Output_/dense/bias/Assign3^Output_/dense/bias/ExponentialMovingAverage/Assign!^Output_/dense/kernel/Adam/Assign#^Output_/dense/kernel/Adam_1/Assign^Output_/dense/kernel/Assign5^Output_/dense/kernel/ExponentialMovingAverage/Assign-^Reconstruction_Output/dense/bias/Adam/Assign/^Reconstruction_Output/dense/bias/Adam_1/Assign(^Reconstruction_Output/dense/bias/AssignA^Reconstruction_Output/dense/bias/ExponentialMovingAverage/Assign/^Reconstruction_Output/dense/kernel/Adam/Assign1^Reconstruction_Output/dense/kernel/Adam_1/Assign*^Reconstruction_Output/dense/kernel/AssignC^Reconstruction_Output/dense/kernel/ExponentialMovingAverage/Assign^beta1_power/Assign^beta2_power/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense/bias/Assign+^dense/bias/ExponentialMovingAverage/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign-^dense/kernel/ExponentialMovingAverage/Assign"/device:GPU:1
&
initNoOp
^init/NoOp^init/NoOp_1
[
save_1/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
dtype0*
_output_shapes
: 
?
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_b51ffb45488347bba7671c628f43edda/part*
dtype0*
_output_shapes
: 
j
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: 
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
?
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)BackupVariables/CCN_1Conv_x0/convA10/biasB+BackupVariables/CCN_1Conv_x0/convA10/kernelB)BackupVariables/CCN_1Conv_x0/convA11/biasB+BackupVariables/CCN_1Conv_x0/convA11/kernelB)BackupVariables/CCN_1Conv_x0/convB10/biasB+BackupVariables/CCN_1Conv_x0/convB10/kernelB)BackupVariables/CCN_1Conv_x0/convB11/biasB+BackupVariables/CCN_1Conv_x0/convB11/kernelB)BackupVariables/CCN_1Conv_x0/convB20/biasB+BackupVariables/CCN_1Conv_x0/convB20/kernelB)BackupVariables/CCN_1Conv_x0/convB21/biasB+BackupVariables/CCN_1Conv_x0/convB21/kernelBBackupVariables/Conv_out__/betaB BackupVariables/Conv_out__/gammaB&BackupVariables/FCU_muiltDense_x0/betaB,BackupVariables/FCU_muiltDense_x0/dense/biasB.BackupVariables/FCU_muiltDense_x0/dense/kernelB'BackupVariables/FCU_muiltDense_x0/gammaB"BackupVariables/Output_/dense/biasB$BackupVariables/Output_/dense/kernelB0BackupVariables/Reconstruction_Output/dense/biasB2BackupVariables/Reconstruction_Output/dense/kernelBBackupVariables/dense/biasBBackupVariables/dense/kernelBglobal_step
?
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?	
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices)BackupVariables/CCN_1Conv_x0/convA10/bias+BackupVariables/CCN_1Conv_x0/convA10/kernel)BackupVariables/CCN_1Conv_x0/convA11/bias+BackupVariables/CCN_1Conv_x0/convA11/kernel)BackupVariables/CCN_1Conv_x0/convB10/bias+BackupVariables/CCN_1Conv_x0/convB10/kernel)BackupVariables/CCN_1Conv_x0/convB11/bias+BackupVariables/CCN_1Conv_x0/convB11/kernel)BackupVariables/CCN_1Conv_x0/convB20/bias+BackupVariables/CCN_1Conv_x0/convB20/kernel)BackupVariables/CCN_1Conv_x0/convB21/bias+BackupVariables/CCN_1Conv_x0/convB21/kernelBackupVariables/Conv_out__/beta BackupVariables/Conv_out__/gamma&BackupVariables/FCU_muiltDense_x0/beta,BackupVariables/FCU_muiltDense_x0/dense/bias.BackupVariables/FCU_muiltDense_x0/dense/kernel'BackupVariables/FCU_muiltDense_x0/gamma"BackupVariables/Output_/dense/bias$BackupVariables/Output_/dense/kernel0BackupVariables/Reconstruction_Output/dense/bias2BackupVariables/Reconstruction_Output/dense/kernelBackupVariables/dense/biasBackupVariables/dense/kernelglobal_step"/device:CPU:0*'
dtypes
2
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*)
_class
loc:@save_1/ShardedFilename
o
save_1/ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0
?
save_1/ShardedFilename_1ShardedFilenamesave_1/StringJoinsave_1/ShardedFilename_1/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
?
save_1/SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:b*?
value?B?bBCCN_1Conv_x0/convA10/biasBCCN_1Conv_x0/convA10/bias/AdamB CCN_1Conv_x0/convA10/bias/Adam_1B2CCN_1Conv_x0/convA10/bias/ExponentialMovingAverageBCCN_1Conv_x0/convA10/kernelB CCN_1Conv_x0/convA10/kernel/AdamB"CCN_1Conv_x0/convA10/kernel/Adam_1B4CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convA11/biasBCCN_1Conv_x0/convA11/bias/AdamB CCN_1Conv_x0/convA11/bias/Adam_1B2CCN_1Conv_x0/convA11/bias/ExponentialMovingAverageBCCN_1Conv_x0/convA11/kernelB CCN_1Conv_x0/convA11/kernel/AdamB"CCN_1Conv_x0/convA11/kernel/Adam_1B4CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB10/biasBCCN_1Conv_x0/convB10/bias/AdamB CCN_1Conv_x0/convB10/bias/Adam_1B2CCN_1Conv_x0/convB10/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB10/kernelB CCN_1Conv_x0/convB10/kernel/AdamB"CCN_1Conv_x0/convB10/kernel/Adam_1B4CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB11/biasBCCN_1Conv_x0/convB11/bias/AdamB CCN_1Conv_x0/convB11/bias/Adam_1B2CCN_1Conv_x0/convB11/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB11/kernelB CCN_1Conv_x0/convB11/kernel/AdamB"CCN_1Conv_x0/convB11/kernel/Adam_1B4CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB20/biasBCCN_1Conv_x0/convB20/bias/AdamB CCN_1Conv_x0/convB20/bias/Adam_1B2CCN_1Conv_x0/convB20/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB20/kernelB CCN_1Conv_x0/convB20/kernel/AdamB"CCN_1Conv_x0/convB20/kernel/Adam_1B4CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB21/biasBCCN_1Conv_x0/convB21/bias/AdamB CCN_1Conv_x0/convB21/bias/Adam_1B2CCN_1Conv_x0/convB21/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB21/kernelB CCN_1Conv_x0/convB21/kernel/AdamB"CCN_1Conv_x0/convB21/kernel/Adam_1B4CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverageBConv_out__/betaBConv_out__/beta/AdamBConv_out__/beta/Adam_1B(Conv_out__/beta/ExponentialMovingAverageBConv_out__/gammaBConv_out__/gamma/AdamBConv_out__/gamma/Adam_1B)Conv_out__/gamma/ExponentialMovingAverageBFCU_muiltDense_x0/betaBFCU_muiltDense_x0/beta/AdamBFCU_muiltDense_x0/beta/Adam_1B/FCU_muiltDense_x0/beta/ExponentialMovingAverageBFCU_muiltDense_x0/dense/biasB!FCU_muiltDense_x0/dense/bias/AdamB#FCU_muiltDense_x0/dense/bias/Adam_1B5FCU_muiltDense_x0/dense/bias/ExponentialMovingAverageBFCU_muiltDense_x0/dense/kernelB#FCU_muiltDense_x0/dense/kernel/AdamB%FCU_muiltDense_x0/dense/kernel/Adam_1B7FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverageBFCU_muiltDense_x0/gammaBFCU_muiltDense_x0/gamma/AdamBFCU_muiltDense_x0/gamma/Adam_1B0FCU_muiltDense_x0/gamma/ExponentialMovingAverageBOutput_/dense/biasBOutput_/dense/bias/AdamBOutput_/dense/bias/Adam_1B+Output_/dense/bias/ExponentialMovingAverageBOutput_/dense/kernelBOutput_/dense/kernel/AdamBOutput_/dense/kernel/Adam_1B-Output_/dense/kernel/ExponentialMovingAverageB Reconstruction_Output/dense/biasB%Reconstruction_Output/dense/bias/AdamB'Reconstruction_Output/dense/bias/Adam_1B9Reconstruction_Output/dense/bias/ExponentialMovingAverageB"Reconstruction_Output/dense/kernelB'Reconstruction_Output/dense/kernel/AdamB)Reconstruction_Output/dense/kernel/Adam_1B;Reconstruction_Output/dense/kernel/ExponentialMovingAverageBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1B#dense/bias/ExponentialMovingAverageBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1B%dense/kernel/ExponentialMovingAverage
?
 save_1/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*?
value?B?bB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:b
?
save_1/SaveV2_1SaveV2save_1/ShardedFilename_1save_1/SaveV2_1/tensor_names save_1/SaveV2_1/shape_and_slicesCCN_1Conv_x0/convA10/biasCCN_1Conv_x0/convA10/bias/Adam CCN_1Conv_x0/convA10/bias/Adam_12CCN_1Conv_x0/convA10/bias/ExponentialMovingAverageCCN_1Conv_x0/convA10/kernel CCN_1Conv_x0/convA10/kernel/Adam"CCN_1Conv_x0/convA10/kernel/Adam_14CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverageCCN_1Conv_x0/convA11/biasCCN_1Conv_x0/convA11/bias/Adam CCN_1Conv_x0/convA11/bias/Adam_12CCN_1Conv_x0/convA11/bias/ExponentialMovingAverageCCN_1Conv_x0/convA11/kernel CCN_1Conv_x0/convA11/kernel/Adam"CCN_1Conv_x0/convA11/kernel/Adam_14CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverageCCN_1Conv_x0/convB10/biasCCN_1Conv_x0/convB10/bias/Adam CCN_1Conv_x0/convB10/bias/Adam_12CCN_1Conv_x0/convB10/bias/ExponentialMovingAverageCCN_1Conv_x0/convB10/kernel CCN_1Conv_x0/convB10/kernel/Adam"CCN_1Conv_x0/convB10/kernel/Adam_14CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverageCCN_1Conv_x0/convB11/biasCCN_1Conv_x0/convB11/bias/Adam CCN_1Conv_x0/convB11/bias/Adam_12CCN_1Conv_x0/convB11/bias/ExponentialMovingAverageCCN_1Conv_x0/convB11/kernel CCN_1Conv_x0/convB11/kernel/Adam"CCN_1Conv_x0/convB11/kernel/Adam_14CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverageCCN_1Conv_x0/convB20/biasCCN_1Conv_x0/convB20/bias/Adam CCN_1Conv_x0/convB20/bias/Adam_12CCN_1Conv_x0/convB20/bias/ExponentialMovingAverageCCN_1Conv_x0/convB20/kernel CCN_1Conv_x0/convB20/kernel/Adam"CCN_1Conv_x0/convB20/kernel/Adam_14CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverageCCN_1Conv_x0/convB21/biasCCN_1Conv_x0/convB21/bias/Adam CCN_1Conv_x0/convB21/bias/Adam_12CCN_1Conv_x0/convB21/bias/ExponentialMovingAverageCCN_1Conv_x0/convB21/kernel CCN_1Conv_x0/convB21/kernel/Adam"CCN_1Conv_x0/convB21/kernel/Adam_14CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverageConv_out__/betaConv_out__/beta/AdamConv_out__/beta/Adam_1(Conv_out__/beta/ExponentialMovingAverageConv_out__/gammaConv_out__/gamma/AdamConv_out__/gamma/Adam_1)Conv_out__/gamma/ExponentialMovingAverageFCU_muiltDense_x0/betaFCU_muiltDense_x0/beta/AdamFCU_muiltDense_x0/beta/Adam_1/FCU_muiltDense_x0/beta/ExponentialMovingAverageFCU_muiltDense_x0/dense/bias!FCU_muiltDense_x0/dense/bias/Adam#FCU_muiltDense_x0/dense/bias/Adam_15FCU_muiltDense_x0/dense/bias/ExponentialMovingAverageFCU_muiltDense_x0/dense/kernel#FCU_muiltDense_x0/dense/kernel/Adam%FCU_muiltDense_x0/dense/kernel/Adam_17FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverageFCU_muiltDense_x0/gammaFCU_muiltDense_x0/gamma/AdamFCU_muiltDense_x0/gamma/Adam_10FCU_muiltDense_x0/gamma/ExponentialMovingAverageOutput_/dense/biasOutput_/dense/bias/AdamOutput_/dense/bias/Adam_1+Output_/dense/bias/ExponentialMovingAverageOutput_/dense/kernelOutput_/dense/kernel/AdamOutput_/dense/kernel/Adam_1-Output_/dense/kernel/ExponentialMovingAverage Reconstruction_Output/dense/bias%Reconstruction_Output/dense/bias/Adam'Reconstruction_Output/dense/bias/Adam_19Reconstruction_Output/dense/bias/ExponentialMovingAverage"Reconstruction_Output/dense/kernel'Reconstruction_Output/dense/kernel/Adam)Reconstruction_Output/dense/kernel/Adam_1;Reconstruction_Output/dense/kernel/ExponentialMovingAveragebeta1_powerbeta2_power
dense/biasdense/bias/Adamdense/bias/Adam_1#dense/bias/ExponentialMovingAveragedense/kerneldense/kernel/Adamdense/kernel/Adam_1%dense/kernel/ExponentialMovingAverage"/device:CPU:0*p
dtypesf
d2b
?
save_1/control_dependency_1Identitysave_1/ShardedFilename_1^save_1/SaveV2_1"/device:CPU:0*+
_class!
loc:@save_1/ShardedFilename_1*
_output_shapes
: *
T0
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilenamesave_1/ShardedFilename_1^save_1/control_dependency^save_1/control_dependency_1"/device:CPU:0*
_output_shapes
:*
T0*
N
{
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency^save_1/control_dependency_1"/device:CPU:0*
_output_shapes
: *
T0
?
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)BackupVariables/CCN_1Conv_x0/convA10/biasB+BackupVariables/CCN_1Conv_x0/convA10/kernelB)BackupVariables/CCN_1Conv_x0/convA11/biasB+BackupVariables/CCN_1Conv_x0/convA11/kernelB)BackupVariables/CCN_1Conv_x0/convB10/biasB+BackupVariables/CCN_1Conv_x0/convB10/kernelB)BackupVariables/CCN_1Conv_x0/convB11/biasB+BackupVariables/CCN_1Conv_x0/convB11/kernelB)BackupVariables/CCN_1Conv_x0/convB20/biasB+BackupVariables/CCN_1Conv_x0/convB20/kernelB)BackupVariables/CCN_1Conv_x0/convB21/biasB+BackupVariables/CCN_1Conv_x0/convB21/kernelBBackupVariables/Conv_out__/betaB BackupVariables/Conv_out__/gammaB&BackupVariables/FCU_muiltDense_x0/betaB,BackupVariables/FCU_muiltDense_x0/dense/biasB.BackupVariables/FCU_muiltDense_x0/dense/kernelB'BackupVariables/FCU_muiltDense_x0/gammaB"BackupVariables/Output_/dense/biasB$BackupVariables/Output_/dense/kernelB0BackupVariables/Reconstruction_Output/dense/biasB2BackupVariables/Reconstruction_Output/dense/kernelBBackupVariables/dense/biasBBackupVariables/dense/kernelBglobal_step
?
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*'
dtypes
2*x
_output_shapesf
d:::::::::::::::::::::::::
?
save_1/AssignAssign)BackupVariables/CCN_1Conv_x0/convA10/biassave_1/RestoreV2"/device:CPU:0*
T0*
_output_shapes	
:?*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convA10/bias
?
save_1/Assign_1Assign+BackupVariables/CCN_1Conv_x0/convA10/kernelsave_1/RestoreV2:1"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convA10/kernel*
T0*#
_output_shapes
:?
?
save_1/Assign_2Assign)BackupVariables/CCN_1Conv_x0/convA11/biassave_1/RestoreV2:2"/device:CPU:0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convA11/bias*
T0*
_output_shapes	
:?
?
save_1/Assign_3Assign+BackupVariables/CCN_1Conv_x0/convA11/kernelsave_1/RestoreV2:3"/device:CPU:0*$
_output_shapes
:??*
T0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convA11/kernel
?
save_1/Assign_4Assign)BackupVariables/CCN_1Conv_x0/convB10/biassave_1/RestoreV2:4"/device:CPU:0*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB10/bias*
_output_shapes	
:?
?
save_1/Assign_5Assign+BackupVariables/CCN_1Conv_x0/convB10/kernelsave_1/RestoreV2:5"/device:CPU:0*
T0*$
_output_shapes
:??*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB10/kernel
?
save_1/Assign_6Assign)BackupVariables/CCN_1Conv_x0/convB11/biassave_1/RestoreV2:6"/device:CPU:0*
_output_shapes	
:?*
T0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB11/bias
?
save_1/Assign_7Assign+BackupVariables/CCN_1Conv_x0/convB11/kernelsave_1/RestoreV2:7"/device:CPU:0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB11/kernel*
T0*$
_output_shapes
:??
?
save_1/Assign_8Assign)BackupVariables/CCN_1Conv_x0/convB20/biassave_1/RestoreV2:8"/device:CPU:0*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB20/bias*
_output_shapes	
:?*
T0
?
save_1/Assign_9Assign+BackupVariables/CCN_1Conv_x0/convB20/kernelsave_1/RestoreV2:9"/device:CPU:0*
T0*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??
?
save_1/Assign_10Assign)BackupVariables/CCN_1Conv_x0/convB21/biassave_1/RestoreV2:10"/device:CPU:0*
_output_shapes	
:?*<
_class2
0.loc:@BackupVariables/CCN_1Conv_x0/convB21/bias*
T0
?
save_1/Assign_11Assign+BackupVariables/CCN_1Conv_x0/convB21/kernelsave_1/RestoreV2:11"/device:CPU:0*$
_output_shapes
:??*>
_class4
20loc:@BackupVariables/CCN_1Conv_x0/convB21/kernel*
T0
?
save_1/Assign_12AssignBackupVariables/Conv_out__/betasave_1/RestoreV2:12"/device:CPU:0*
_output_shapes	
:?*
T0*2
_class(
&$loc:@BackupVariables/Conv_out__/beta
?
save_1/Assign_13Assign BackupVariables/Conv_out__/gammasave_1/RestoreV2:13"/device:CPU:0*
_output_shapes	
:?*
T0*3
_class)
'%loc:@BackupVariables/Conv_out__/gamma
?
save_1/Assign_14Assign&BackupVariables/FCU_muiltDense_x0/betasave_1/RestoreV2:14"/device:CPU:0*
T0*9
_class/
-+loc:@BackupVariables/FCU_muiltDense_x0/beta*
_output_shapes
: 
?
save_1/Assign_15Assign,BackupVariables/FCU_muiltDense_x0/dense/biassave_1/RestoreV2:15"/device:CPU:0*
T0*?
_class5
31loc:@BackupVariables/FCU_muiltDense_x0/dense/bias*
_output_shapes
: 
?
save_1/Assign_16Assign.BackupVariables/FCU_muiltDense_x0/dense/kernelsave_1/RestoreV2:16"/device:CPU:0*
_output_shapes

:  *A
_class7
53loc:@BackupVariables/FCU_muiltDense_x0/dense/kernel*
T0
?
save_1/Assign_17Assign'BackupVariables/FCU_muiltDense_x0/gammasave_1/RestoreV2:17"/device:CPU:0*:
_class0
.,loc:@BackupVariables/FCU_muiltDense_x0/gamma*
_output_shapes
: *
T0
?
save_1/Assign_18Assign"BackupVariables/Output_/dense/biassave_1/RestoreV2:18"/device:CPU:0*5
_class+
)'loc:@BackupVariables/Output_/dense/bias*
_output_shapes
:*
T0
?
save_1/Assign_19Assign$BackupVariables/Output_/dense/kernelsave_1/RestoreV2:19"/device:CPU:0*
_output_shapes

: *
T0*7
_class-
+)loc:@BackupVariables/Output_/dense/kernel
?
save_1/Assign_20Assign0BackupVariables/Reconstruction_Output/dense/biassave_1/RestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:*C
_class9
75loc:@BackupVariables/Reconstruction_Output/dense/bias
?
save_1/Assign_21Assign2BackupVariables/Reconstruction_Output/dense/kernelsave_1/RestoreV2:21"/device:CPU:0*E
_class;
97loc:@BackupVariables/Reconstruction_Output/dense/kernel*
T0*
_output_shapes
:	?
?
save_1/Assign_22AssignBackupVariables/dense/biassave_1/RestoreV2:22"/device:CPU:0*-
_class#
!loc:@BackupVariables/dense/bias*
_output_shapes
: *
T0
?
save_1/Assign_23AssignBackupVariables/dense/kernelsave_1/RestoreV2:23"/device:CPU:0*
_output_shapes
:	? *
T0*/
_class%
#!loc:@BackupVariables/dense/kernel
?
save_1/Assign_24Assignglobal_stepsave_1/RestoreV2:24"/device:CPU:0*
_output_shapes
: *
_class
loc:@global_step*
T0
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9"/device:CPU:0
?
save_1/RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:b*?
value?B?bBCCN_1Conv_x0/convA10/biasBCCN_1Conv_x0/convA10/bias/AdamB CCN_1Conv_x0/convA10/bias/Adam_1B2CCN_1Conv_x0/convA10/bias/ExponentialMovingAverageBCCN_1Conv_x0/convA10/kernelB CCN_1Conv_x0/convA10/kernel/AdamB"CCN_1Conv_x0/convA10/kernel/Adam_1B4CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convA11/biasBCCN_1Conv_x0/convA11/bias/AdamB CCN_1Conv_x0/convA11/bias/Adam_1B2CCN_1Conv_x0/convA11/bias/ExponentialMovingAverageBCCN_1Conv_x0/convA11/kernelB CCN_1Conv_x0/convA11/kernel/AdamB"CCN_1Conv_x0/convA11/kernel/Adam_1B4CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB10/biasBCCN_1Conv_x0/convB10/bias/AdamB CCN_1Conv_x0/convB10/bias/Adam_1B2CCN_1Conv_x0/convB10/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB10/kernelB CCN_1Conv_x0/convB10/kernel/AdamB"CCN_1Conv_x0/convB10/kernel/Adam_1B4CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB11/biasBCCN_1Conv_x0/convB11/bias/AdamB CCN_1Conv_x0/convB11/bias/Adam_1B2CCN_1Conv_x0/convB11/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB11/kernelB CCN_1Conv_x0/convB11/kernel/AdamB"CCN_1Conv_x0/convB11/kernel/Adam_1B4CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB20/biasBCCN_1Conv_x0/convB20/bias/AdamB CCN_1Conv_x0/convB20/bias/Adam_1B2CCN_1Conv_x0/convB20/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB20/kernelB CCN_1Conv_x0/convB20/kernel/AdamB"CCN_1Conv_x0/convB20/kernel/Adam_1B4CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverageBCCN_1Conv_x0/convB21/biasBCCN_1Conv_x0/convB21/bias/AdamB CCN_1Conv_x0/convB21/bias/Adam_1B2CCN_1Conv_x0/convB21/bias/ExponentialMovingAverageBCCN_1Conv_x0/convB21/kernelB CCN_1Conv_x0/convB21/kernel/AdamB"CCN_1Conv_x0/convB21/kernel/Adam_1B4CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverageBConv_out__/betaBConv_out__/beta/AdamBConv_out__/beta/Adam_1B(Conv_out__/beta/ExponentialMovingAverageBConv_out__/gammaBConv_out__/gamma/AdamBConv_out__/gamma/Adam_1B)Conv_out__/gamma/ExponentialMovingAverageBFCU_muiltDense_x0/betaBFCU_muiltDense_x0/beta/AdamBFCU_muiltDense_x0/beta/Adam_1B/FCU_muiltDense_x0/beta/ExponentialMovingAverageBFCU_muiltDense_x0/dense/biasB!FCU_muiltDense_x0/dense/bias/AdamB#FCU_muiltDense_x0/dense/bias/Adam_1B5FCU_muiltDense_x0/dense/bias/ExponentialMovingAverageBFCU_muiltDense_x0/dense/kernelB#FCU_muiltDense_x0/dense/kernel/AdamB%FCU_muiltDense_x0/dense/kernel/Adam_1B7FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverageBFCU_muiltDense_x0/gammaBFCU_muiltDense_x0/gamma/AdamBFCU_muiltDense_x0/gamma/Adam_1B0FCU_muiltDense_x0/gamma/ExponentialMovingAverageBOutput_/dense/biasBOutput_/dense/bias/AdamBOutput_/dense/bias/Adam_1B+Output_/dense/bias/ExponentialMovingAverageBOutput_/dense/kernelBOutput_/dense/kernel/AdamBOutput_/dense/kernel/Adam_1B-Output_/dense/kernel/ExponentialMovingAverageB Reconstruction_Output/dense/biasB%Reconstruction_Output/dense/bias/AdamB'Reconstruction_Output/dense/bias/Adam_1B9Reconstruction_Output/dense/bias/ExponentialMovingAverageB"Reconstruction_Output/dense/kernelB'Reconstruction_Output/dense/kernel/AdamB)Reconstruction_Output/dense/kernel/Adam_1B;Reconstruction_Output/dense/kernel/ExponentialMovingAverageBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1B#dense/bias/ExponentialMovingAverageBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1B%dense/kernel/ExponentialMovingAverage
?
#save_1/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*?
value?B?bB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:b*
dtype0
?
save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*p
dtypesf
d2b
?
save_1/Assign_25AssignCCN_1Conv_x0/convA10/biassave_1/RestoreV2_1"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
save_1/Assign_26AssignCCN_1Conv_x0/convA10/bias/Adamsave_1/RestoreV2_1:1"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?*
T0
?
save_1/Assign_27Assign CCN_1Conv_x0/convA10/bias/Adam_1save_1/RestoreV2_1:2"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes	
:?*
T0
?
save_1/Assign_28Assign2CCN_1Conv_x0/convA10/bias/ExponentialMovingAveragesave_1/RestoreV2_1:3"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias
?
save_1/Assign_29AssignCCN_1Conv_x0/convA10/kernelsave_1/RestoreV2_1:4"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*#
_output_shapes
:?
?
save_1/Assign_30Assign CCN_1Conv_x0/convA10/kernel/Adamsave_1/RestoreV2_1:5"/device:GPU:1*
T0*#
_output_shapes
:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
save_1/Assign_31Assign"CCN_1Conv_x0/convA10/kernel/Adam_1save_1/RestoreV2_1:6"/device:GPU:1*
T0*#
_output_shapes
:?*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel
?
save_1/Assign_32Assign4CCN_1Conv_x0/convA10/kernel/ExponentialMovingAveragesave_1/RestoreV2_1:7"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA10/kernel*
T0*#
_output_shapes
:?
?
save_1/Assign_33AssignCCN_1Conv_x0/convA11/biassave_1/RestoreV2_1:8"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0
?
save_1/Assign_34AssignCCN_1Conv_x0/convA11/bias/Adamsave_1/RestoreV2_1:9"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0
?
save_1/Assign_35Assign CCN_1Conv_x0/convA11/bias/Adam_1save_1/RestoreV2_1:10"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias
?
save_1/Assign_36Assign2CCN_1Conv_x0/convA11/bias/ExponentialMovingAveragesave_1/RestoreV2_1:11"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convA11/bias*
T0*
_output_shapes	
:?
?
save_1/Assign_37AssignCCN_1Conv_x0/convA11/kernelsave_1/RestoreV2_1:12"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel
?
save_1/Assign_38Assign CCN_1Conv_x0/convA11/kernel/Adamsave_1/RestoreV2_1:13"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*
T0
?
save_1/Assign_39Assign"CCN_1Conv_x0/convA11/kernel/Adam_1save_1/RestoreV2_1:14"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??*
T0
?
save_1/Assign_40Assign4CCN_1Conv_x0/convA11/kernel/ExponentialMovingAveragesave_1/RestoreV2_1:15"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convA11/kernel*$
_output_shapes
:??*
T0
?
save_1/Assign_41AssignCCN_1Conv_x0/convB10/biassave_1/RestoreV2_1:16"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
T0
?
save_1/Assign_42AssignCCN_1Conv_x0/convB10/bias/Adamsave_1/RestoreV2_1:17"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
?
save_1/Assign_43Assign CCN_1Conv_x0/convB10/bias/Adam_1save_1/RestoreV2_1:18"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias
?
save_1/Assign_44Assign2CCN_1Conv_x0/convB10/bias/ExponentialMovingAveragesave_1/RestoreV2_1:19"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB10/bias*
_output_shapes	
:?
?
save_1/Assign_45AssignCCN_1Conv_x0/convB10/kernelsave_1/RestoreV2_1:20"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0
?
save_1/Assign_46Assign CCN_1Conv_x0/convB10/kernel/Adamsave_1/RestoreV2_1:21"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*
T0*$
_output_shapes
:??
?
save_1/Assign_47Assign"CCN_1Conv_x0/convB10/kernel/Adam_1save_1/RestoreV2_1:22"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel*$
_output_shapes
:??*
T0
?
save_1/Assign_48Assign4CCN_1Conv_x0/convB10/kernel/ExponentialMovingAveragesave_1/RestoreV2_1:23"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB10/kernel
?
save_1/Assign_49AssignCCN_1Conv_x0/convB11/biassave_1/RestoreV2_1:24"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?*
T0
?
save_1/Assign_50AssignCCN_1Conv_x0/convB11/bias/Adamsave_1/RestoreV2_1:25"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?*
T0
?
save_1/Assign_51Assign CCN_1Conv_x0/convB11/bias/Adam_1save_1/RestoreV2_1:26"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?*
T0
?
save_1/Assign_52Assign2CCN_1Conv_x0/convB11/bias/ExponentialMovingAveragesave_1/RestoreV2_1:27"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB11/bias*
_output_shapes	
:?*
T0
?
save_1/Assign_53AssignCCN_1Conv_x0/convB11/kernelsave_1/RestoreV2_1:28"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*$
_output_shapes
:??*
T0
?
save_1/Assign_54Assign CCN_1Conv_x0/convB11/kernel/Adamsave_1/RestoreV2_1:29"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*$
_output_shapes
:??*
T0
?
save_1/Assign_55Assign"CCN_1Conv_x0/convB11/kernel/Adam_1save_1/RestoreV2_1:30"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel*
T0*$
_output_shapes
:??
?
save_1/Assign_56Assign4CCN_1Conv_x0/convB11/kernel/ExponentialMovingAveragesave_1/RestoreV2_1:31"/device:GPU:1*$
_output_shapes
:??*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB11/kernel
?
save_1/Assign_57AssignCCN_1Conv_x0/convB20/biassave_1/RestoreV2_1:32"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
save_1/Assign_58AssignCCN_1Conv_x0/convB20/bias/Adamsave_1/RestoreV2_1:33"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
save_1/Assign_59Assign CCN_1Conv_x0/convB20/bias/Adam_1save_1/RestoreV2_1:34"/device:GPU:1*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias*
T0
?
save_1/Assign_60Assign2CCN_1Conv_x0/convB20/bias/ExponentialMovingAveragesave_1/RestoreV2_1:35"/device:GPU:1*
_output_shapes	
:?*
T0*,
_class"
 loc:@CCN_1Conv_x0/convB20/bias
?
save_1/Assign_61AssignCCN_1Conv_x0/convB20/kernelsave_1/RestoreV2_1:36"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0
?
save_1/Assign_62Assign CCN_1Conv_x0/convB20/kernel/Adamsave_1/RestoreV2_1:37"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??
?
save_1/Assign_63Assign"CCN_1Conv_x0/convB20/kernel/Adam_1save_1/RestoreV2_1:38"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*
T0*$
_output_shapes
:??
?
save_1/Assign_64Assign4CCN_1Conv_x0/convB20/kernel/ExponentialMovingAveragesave_1/RestoreV2_1:39"/device:GPU:1*.
_class$
" loc:@CCN_1Conv_x0/convB20/kernel*$
_output_shapes
:??*
T0
?
save_1/Assign_65AssignCCN_1Conv_x0/convB21/biassave_1/RestoreV2_1:40"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes	
:?*
T0
?
save_1/Assign_66AssignCCN_1Conv_x0/convB21/bias/Adamsave_1/RestoreV2_1:41"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
T0*
_output_shapes	
:?
?
save_1/Assign_67Assign CCN_1Conv_x0/convB21/bias/Adam_1save_1/RestoreV2_1:42"/device:GPU:1*
T0*
_output_shapes	
:?*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias
?
save_1/Assign_68Assign2CCN_1Conv_x0/convB21/bias/ExponentialMovingAveragesave_1/RestoreV2_1:43"/device:GPU:1*,
_class"
 loc:@CCN_1Conv_x0/convB21/bias*
_output_shapes	
:?*
T0
?
save_1/Assign_69AssignCCN_1Conv_x0/convB21/kernelsave_1/RestoreV2_1:44"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
T0
?
save_1/Assign_70Assign CCN_1Conv_x0/convB21/kernel/Adamsave_1/RestoreV2_1:45"/device:GPU:1*
T0*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*$
_output_shapes
:??
?
save_1/Assign_71Assign"CCN_1Conv_x0/convB21/kernel/Adam_1save_1/RestoreV2_1:46"/device:GPU:1*
T0*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel
?
save_1/Assign_72Assign4CCN_1Conv_x0/convB21/kernel/ExponentialMovingAveragesave_1/RestoreV2_1:47"/device:GPU:1*$
_output_shapes
:??*.
_class$
" loc:@CCN_1Conv_x0/convB21/kernel*
T0
?
save_1/Assign_73AssignConv_out__/betasave_1/RestoreV2_1:48"/device:GPU:1*
T0*
_output_shapes	
:?*"
_class
loc:@Conv_out__/beta
?
save_1/Assign_74AssignConv_out__/beta/Adamsave_1/RestoreV2_1:49"/device:GPU:1*
_output_shapes	
:?*
T0*"
_class
loc:@Conv_out__/beta
?
save_1/Assign_75AssignConv_out__/beta/Adam_1save_1/RestoreV2_1:50"/device:GPU:1*
_output_shapes	
:?*
T0*"
_class
loc:@Conv_out__/beta
?
save_1/Assign_76Assign(Conv_out__/beta/ExponentialMovingAveragesave_1/RestoreV2_1:51"/device:GPU:1*
_output_shapes	
:?*
T0*"
_class
loc:@Conv_out__/beta
?
save_1/Assign_77AssignConv_out__/gammasave_1/RestoreV2_1:52"/device:GPU:1*
_output_shapes	
:?*
T0*#
_class
loc:@Conv_out__/gamma
?
save_1/Assign_78AssignConv_out__/gamma/Adamsave_1/RestoreV2_1:53"/device:GPU:1*
T0*
_output_shapes	
:?*#
_class
loc:@Conv_out__/gamma
?
save_1/Assign_79AssignConv_out__/gamma/Adam_1save_1/RestoreV2_1:54"/device:GPU:1*
_output_shapes	
:?*
T0*#
_class
loc:@Conv_out__/gamma
?
save_1/Assign_80Assign)Conv_out__/gamma/ExponentialMovingAveragesave_1/RestoreV2_1:55"/device:GPU:1*#
_class
loc:@Conv_out__/gamma*
_output_shapes	
:?*
T0
?
save_1/Assign_81AssignFCU_muiltDense_x0/betasave_1/RestoreV2_1:56"/device:GPU:1*
_output_shapes
: *)
_class
loc:@FCU_muiltDense_x0/beta*
T0
?
save_1/Assign_82AssignFCU_muiltDense_x0/beta/Adamsave_1/RestoreV2_1:57"/device:GPU:1*
_output_shapes
: *)
_class
loc:@FCU_muiltDense_x0/beta*
T0
?
save_1/Assign_83AssignFCU_muiltDense_x0/beta/Adam_1save_1/RestoreV2_1:58"/device:GPU:1*
T0*)
_class
loc:@FCU_muiltDense_x0/beta*
_output_shapes
: 
?
save_1/Assign_84Assign/FCU_muiltDense_x0/beta/ExponentialMovingAveragesave_1/RestoreV2_1:59"/device:GPU:1*)
_class
loc:@FCU_muiltDense_x0/beta*
_output_shapes
: *
T0
?
save_1/Assign_85AssignFCU_muiltDense_x0/dense/biassave_1/RestoreV2_1:60"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0*
_output_shapes
: 
?
save_1/Assign_86Assign!FCU_muiltDense_x0/dense/bias/Adamsave_1/RestoreV2_1:61"/device:GPU:1*
T0*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
_output_shapes
: 
?
save_1/Assign_87Assign#FCU_muiltDense_x0/dense/bias/Adam_1save_1/RestoreV2_1:62"/device:GPU:1*/
_class%
#!loc:@FCU_muiltDense_x0/dense/bias*
T0*
_output_shapes
: 
?
save_1/Assign_88Assign5FCU_muiltDense_x0/dense/bias/ExponentialMovingAveragesave_1/RestoreV2_1:63"/device:GPU:1*
T0*
_output_shapes
: */
_class%
#!loc:@FCU_muiltDense_x0/dense/bias
?
save_1/Assign_89AssignFCU_muiltDense_x0/dense/kernelsave_1/RestoreV2_1:64"/device:GPU:1*
_output_shapes

:  *
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
save_1/Assign_90Assign#FCU_muiltDense_x0/dense/kernel/Adamsave_1/RestoreV2_1:65"/device:GPU:1*
T0*
_output_shapes

:  *1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel
?
save_1/Assign_91Assign%FCU_muiltDense_x0/dense/kernel/Adam_1save_1/RestoreV2_1:66"/device:GPU:1*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
T0*
_output_shapes

:  
?
save_1/Assign_92Assign7FCU_muiltDense_x0/dense/kernel/ExponentialMovingAveragesave_1/RestoreV2_1:67"/device:GPU:1*
T0*1
_class'
%#loc:@FCU_muiltDense_x0/dense/kernel*
_output_shapes

:  
?
save_1/Assign_93AssignFCU_muiltDense_x0/gammasave_1/RestoreV2_1:68"/device:GPU:1**
_class 
loc:@FCU_muiltDense_x0/gamma*
T0*
_output_shapes
: 
?
save_1/Assign_94AssignFCU_muiltDense_x0/gamma/Adamsave_1/RestoreV2_1:69"/device:GPU:1*
T0*
_output_shapes
: **
_class 
loc:@FCU_muiltDense_x0/gamma
?
save_1/Assign_95AssignFCU_muiltDense_x0/gamma/Adam_1save_1/RestoreV2_1:70"/device:GPU:1*
_output_shapes
: *
T0**
_class 
loc:@FCU_muiltDense_x0/gamma
?
save_1/Assign_96Assign0FCU_muiltDense_x0/gamma/ExponentialMovingAveragesave_1/RestoreV2_1:71"/device:GPU:1*
_output_shapes
: *
T0**
_class 
loc:@FCU_muiltDense_x0/gamma
?
save_1/Assign_97AssignOutput_/dense/biassave_1/RestoreV2_1:72"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
_output_shapes
:*
T0
?
save_1/Assign_98AssignOutput_/dense/bias/Adamsave_1/RestoreV2_1:73"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
_output_shapes
:*
T0
?
save_1/Assign_99AssignOutput_/dense/bias/Adam_1save_1/RestoreV2_1:74"/device:GPU:1*
_output_shapes
:*
T0*%
_class
loc:@Output_/dense/bias
?
save_1/Assign_100Assign+Output_/dense/bias/ExponentialMovingAveragesave_1/RestoreV2_1:75"/device:GPU:1*%
_class
loc:@Output_/dense/bias*
T0*
_output_shapes
:
?
save_1/Assign_101AssignOutput_/dense/kernelsave_1/RestoreV2_1:76"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*
_output_shapes

: *
T0
?
save_1/Assign_102AssignOutput_/dense/kernel/Adamsave_1/RestoreV2_1:77"/device:GPU:1*
T0*'
_class
loc:@Output_/dense/kernel*
_output_shapes

: 
?
save_1/Assign_103AssignOutput_/dense/kernel/Adam_1save_1/RestoreV2_1:78"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*
T0*
_output_shapes

: 
?
save_1/Assign_104Assign-Output_/dense/kernel/ExponentialMovingAveragesave_1/RestoreV2_1:79"/device:GPU:1*'
_class
loc:@Output_/dense/kernel*
T0*
_output_shapes

: 
?
save_1/Assign_105Assign Reconstruction_Output/dense/biassave_1/RestoreV2_1:80"/device:GPU:1*
_output_shapes
:*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
T0
?
save_1/Assign_106Assign%Reconstruction_Output/dense/bias/Adamsave_1/RestoreV2_1:81"/device:GPU:1*
_output_shapes
:*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias
?
save_1/Assign_107Assign'Reconstruction_Output/dense/bias/Adam_1save_1/RestoreV2_1:82"/device:GPU:1*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
T0*
_output_shapes
:
?
save_1/Assign_108Assign9Reconstruction_Output/dense/bias/ExponentialMovingAveragesave_1/RestoreV2_1:83"/device:GPU:1*
T0*3
_class)
'%loc:@Reconstruction_Output/dense/bias*
_output_shapes
:
?
save_1/Assign_109Assign"Reconstruction_Output/dense/kernelsave_1/RestoreV2_1:84"/device:GPU:1*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?
?
save_1/Assign_110Assign'Reconstruction_Output/dense/kernel/Adamsave_1/RestoreV2_1:85"/device:GPU:1*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
T0*
_output_shapes
:	?
?
save_1/Assign_111Assign)Reconstruction_Output/dense/kernel/Adam_1save_1/RestoreV2_1:86"/device:GPU:1*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel*
_output_shapes
:	?
?
save_1/Assign_112Assign;Reconstruction_Output/dense/kernel/ExponentialMovingAveragesave_1/RestoreV2_1:87"/device:GPU:1*
_output_shapes
:	?*
T0*5
_class+
)'loc:@Reconstruction_Output/dense/kernel
?
save_1/Assign_113Assignbeta1_powersave_1/RestoreV2_1:88"/device:GPU:1*
_output_shapes
: *,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
T0
?
save_1/Assign_114Assignbeta2_powersave_1/RestoreV2_1:89"/device:GPU:1*
T0*,
_class"
 loc:@CCN_1Conv_x0/convA10/bias*
_output_shapes
: 
?
save_1/Assign_115Assign
dense/biassave_1/RestoreV2_1:90"/device:GPU:1*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
?
save_1/Assign_116Assigndense/bias/Adamsave_1/RestoreV2_1:91"/device:GPU:1*
T0*
_output_shapes
: *
_class
loc:@dense/bias
?
save_1/Assign_117Assigndense/bias/Adam_1save_1/RestoreV2_1:92"/device:GPU:1*
_output_shapes
: *
_class
loc:@dense/bias*
T0
?
save_1/Assign_118Assign#dense/bias/ExponentialMovingAveragesave_1/RestoreV2_1:93"/device:GPU:1*
T0*
_output_shapes
: *
_class
loc:@dense/bias
?
save_1/Assign_119Assigndense/kernelsave_1/RestoreV2_1:94"/device:GPU:1*
T0*
_output_shapes
:	? *
_class
loc:@dense/kernel
?
save_1/Assign_120Assigndense/kernel/Adamsave_1/RestoreV2_1:95"/device:GPU:1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	? 
?
save_1/Assign_121Assigndense/kernel/Adam_1save_1/RestoreV2_1:96"/device:GPU:1*
_class
loc:@dense/kernel*
T0*
_output_shapes
:	? 
?
save_1/Assign_122Assign%dense/kernel/ExponentialMovingAveragesave_1/RestoreV2_1:97"/device:GPU:1*
_output_shapes
:	? *
T0*
_class
loc:@dense/kernel
?
save_1/restore_shard_1NoOp^save_1/Assign_100^save_1/Assign_101^save_1/Assign_102^save_1/Assign_103^save_1/Assign_104^save_1/Assign_105^save_1/Assign_106^save_1/Assign_107^save_1/Assign_108^save_1/Assign_109^save_1/Assign_110^save_1/Assign_111^save_1/Assign_112^save_1/Assign_113^save_1/Assign_114^save_1/Assign_115^save_1/Assign_116^save_1/Assign_117^save_1/Assign_118^save_1/Assign_119^save_1/Assign_120^save_1/Assign_121^save_1/Assign_122^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_76^save_1/Assign_77^save_1/Assign_78^save_1/Assign_79^save_1/Assign_80^save_1/Assign_81^save_1/Assign_82^save_1/Assign_83^save_1/Assign_84^save_1/Assign_85^save_1/Assign_86^save_1/Assign_87^save_1/Assign_88^save_1/Assign_89^save_1/Assign_90^save_1/Assign_91^save_1/Assign_92^save_1/Assign_93^save_1/Assign_94^save_1/Assign_95^save_1/Assign_96^save_1/Assign_97^save_1/Assign_98^save_1/Assign_99"/device:GPU:1
E
save_1/restore_all/NoOpNoOp^save_1/restore_shard"/device:CPU:0
I
save_1/restore_all/NoOp_1NoOp^save_1/restore_shard_1"/device:GPU:1
P
save_1/restore_allNoOp^save_1/restore_all/NoOp^save_1/restore_all/NoOp_1?
?
f
*__inference_Dataset_map_decode_parse_fn_79

args_0
identity

identity_1

identity_2[
ParseSingleExample/ConstConst*
valueB *
_output_shapes
: *
dtype0]
ParseSingleExample/Const_1Const*
valueB *
_output_shapes
: *
dtype0]
ParseSingleExample/Const_2Const*
dtype0*
valueB *
_output_shapes
: ?
%ParseSingleExample/ParseSingleExampleParseSingleExampleargs_0!ParseSingleExample/Const:output:0#ParseSingleExample/Const_1:output:0#ParseSingleExample/Const_2:output:0*
_output_shapes
: : : *5

dense_keys'
%Intput_data	Name_dataOutput_data*
Tdense
2*
dense_shapes
: : : *
sparse_keys
 *
sparse_types
 *

num_sparse ?
	DecodeRaw	DecodeRaw4ParseSingleExample/ParseSingleExample:dense_values:0*
out_type0*#
_output_shapes
:??????????
DecodeRaw_1	DecodeRaw4ParseSingleExample/ParseSingleExample:dense_values:2*
out_type0*#
_output_shapes
:?????????V
IdentityIdentityDecodeRaw:output:0*
T0*#
_output_shapes
:?????????Z

Identity_1IdentityDecodeRaw_1:output:0*#
_output_shapes
:?????????*
T0m

Identity_2Identity4ParseSingleExample/ParseSingleExample:dense_values:1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*
_input_shapes
: :& "
 
_user_specified_nameargs_0
?
^
-__inference_Dataset_flat_map_read_one_file_64

args_0
identity??TFRecordDatasetQ
compression_typeConst*
_output_shapes
: *
valueB B *
dtype0O
buffer_sizeConst*
dtype0	*
_output_shapes
: *
valueB		 R??s
TFRecordDatasetTFRecordDatasetargs_0compression_type:output:0buffer_size:output:0*
_output_shapes
: a
IdentityIdentityTFRecordDataset:handle:0^TFRecordDataset*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
: 2"
TFRecordDatasetTFRecordDataset:& "
 
_user_specified_nameargs_0"?B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"נ
	variablesȠĠ
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/Const:0
?
CCN_1Conv_x0/convA10/kernel:0"CCN_1Conv_x0/convA10/kernel/Assign"CCN_1Conv_x0/convA10/kernel/read:028CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convA10/bias:0 CCN_1Conv_x0/convA10/bias/Assign CCN_1Conv_x0/convA10/bias/read:02-CCN_1Conv_x0/convA10/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB10/kernel:0"CCN_1Conv_x0/convB10/kernel/Assign"CCN_1Conv_x0/convB10/kernel/read:028CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB10/bias:0 CCN_1Conv_x0/convB10/bias/Assign CCN_1Conv_x0/convB10/bias/read:02-CCN_1Conv_x0/convB10/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB20/kernel:0"CCN_1Conv_x0/convB20/kernel/Assign"CCN_1Conv_x0/convB20/kernel/read:028CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB20/bias:0 CCN_1Conv_x0/convB20/bias/Assign CCN_1Conv_x0/convB20/bias/read:02-CCN_1Conv_x0/convB20/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convA11/kernel:0"CCN_1Conv_x0/convA11/kernel/Assign"CCN_1Conv_x0/convA11/kernel/read:028CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convA11/bias:0 CCN_1Conv_x0/convA11/bias/Assign CCN_1Conv_x0/convA11/bias/read:02-CCN_1Conv_x0/convA11/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB11/kernel:0"CCN_1Conv_x0/convB11/kernel/Assign"CCN_1Conv_x0/convB11/kernel/read:028CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB11/bias:0 CCN_1Conv_x0/convB11/bias/Assign CCN_1Conv_x0/convB11/bias/read:02-CCN_1Conv_x0/convB11/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB21/kernel:0"CCN_1Conv_x0/convB21/kernel/Assign"CCN_1Conv_x0/convB21/kernel/read:028CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB21/bias:0 CCN_1Conv_x0/convB21/bias/Assign CCN_1Conv_x0/convB21/bias/read:02-CCN_1Conv_x0/convB21/bias/Initializer/zeros:08
j
Conv_out__/beta:0Conv_out__/beta/AssignConv_out__/beta/read:02#Conv_out__/beta/Initializer/zeros:08
m
Conv_out__/gamma:0Conv_out__/gamma/AssignConv_out__/gamma/read:02#Conv_out__/gamma/Initializer/ones:08
?
$Reconstruction_Output/dense/kernel:0)Reconstruction_Output/dense/kernel/Assign)Reconstruction_Output/dense/kernel/read:02?Reconstruction_Output/dense/kernel/Initializer/random_uniform:08
?
"Reconstruction_Output/dense/bias:0'Reconstruction_Output/dense/bias/Assign'Reconstruction_Output/dense/bias/read:024Reconstruction_Output/dense/bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
?
 FCU_muiltDense_x0/dense/kernel:0%FCU_muiltDense_x0/dense/kernel/Assign%FCU_muiltDense_x0/dense/kernel/read:02;FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform:08
?
FCU_muiltDense_x0/dense/bias:0#FCU_muiltDense_x0/dense/bias/Assign#FCU_muiltDense_x0/dense/bias/read:020FCU_muiltDense_x0/dense/bias/Initializer/zeros:08
?
FCU_muiltDense_x0/beta:0FCU_muiltDense_x0/beta/AssignFCU_muiltDense_x0/beta/read:02*FCU_muiltDense_x0/beta/Initializer/zeros:08
?
FCU_muiltDense_x0/gamma:0FCU_muiltDense_x0/gamma/AssignFCU_muiltDense_x0/gamma/read:02*FCU_muiltDense_x0/gamma/Initializer/ones:08
?
Output_/dense/kernel:0Output_/dense/kernel/AssignOutput_/dense/kernel/read:021Output_/dense/kernel/Initializer/random_uniform:08
v
Output_/dense/bias:0Output_/dense/bias/AssignOutput_/dense/bias/read:02&Output_/dense/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
?
"CCN_1Conv_x0/convA10/kernel/Adam:0'CCN_1Conv_x0/convA10/kernel/Adam/Assign'CCN_1Conv_x0/convA10/kernel/Adam/read:024CCN_1Conv_x0/convA10/kernel/Adam/Initializer/zeros:0
?
$CCN_1Conv_x0/convA10/kernel/Adam_1:0)CCN_1Conv_x0/convA10/kernel/Adam_1/Assign)CCN_1Conv_x0/convA10/kernel/Adam_1/read:026CCN_1Conv_x0/convA10/kernel/Adam_1/Initializer/zeros:0
?
 CCN_1Conv_x0/convA10/bias/Adam:0%CCN_1Conv_x0/convA10/bias/Adam/Assign%CCN_1Conv_x0/convA10/bias/Adam/read:022CCN_1Conv_x0/convA10/bias/Adam/Initializer/zeros:0
?
"CCN_1Conv_x0/convA10/bias/Adam_1:0'CCN_1Conv_x0/convA10/bias/Adam_1/Assign'CCN_1Conv_x0/convA10/bias/Adam_1/read:024CCN_1Conv_x0/convA10/bias/Adam_1/Initializer/zeros:0
?
"CCN_1Conv_x0/convB10/kernel/Adam:0'CCN_1Conv_x0/convB10/kernel/Adam/Assign'CCN_1Conv_x0/convB10/kernel/Adam/read:024CCN_1Conv_x0/convB10/kernel/Adam/Initializer/zeros:0
?
$CCN_1Conv_x0/convB10/kernel/Adam_1:0)CCN_1Conv_x0/convB10/kernel/Adam_1/Assign)CCN_1Conv_x0/convB10/kernel/Adam_1/read:026CCN_1Conv_x0/convB10/kernel/Adam_1/Initializer/zeros:0
?
 CCN_1Conv_x0/convB10/bias/Adam:0%CCN_1Conv_x0/convB10/bias/Adam/Assign%CCN_1Conv_x0/convB10/bias/Adam/read:022CCN_1Conv_x0/convB10/bias/Adam/Initializer/zeros:0
?
"CCN_1Conv_x0/convB10/bias/Adam_1:0'CCN_1Conv_x0/convB10/bias/Adam_1/Assign'CCN_1Conv_x0/convB10/bias/Adam_1/read:024CCN_1Conv_x0/convB10/bias/Adam_1/Initializer/zeros:0
?
"CCN_1Conv_x0/convB20/kernel/Adam:0'CCN_1Conv_x0/convB20/kernel/Adam/Assign'CCN_1Conv_x0/convB20/kernel/Adam/read:024CCN_1Conv_x0/convB20/kernel/Adam/Initializer/zeros:0
?
$CCN_1Conv_x0/convB20/kernel/Adam_1:0)CCN_1Conv_x0/convB20/kernel/Adam_1/Assign)CCN_1Conv_x0/convB20/kernel/Adam_1/read:026CCN_1Conv_x0/convB20/kernel/Adam_1/Initializer/zeros:0
?
 CCN_1Conv_x0/convB20/bias/Adam:0%CCN_1Conv_x0/convB20/bias/Adam/Assign%CCN_1Conv_x0/convB20/bias/Adam/read:022CCN_1Conv_x0/convB20/bias/Adam/Initializer/zeros:0
?
"CCN_1Conv_x0/convB20/bias/Adam_1:0'CCN_1Conv_x0/convB20/bias/Adam_1/Assign'CCN_1Conv_x0/convB20/bias/Adam_1/read:024CCN_1Conv_x0/convB20/bias/Adam_1/Initializer/zeros:0
?
"CCN_1Conv_x0/convA11/kernel/Adam:0'CCN_1Conv_x0/convA11/kernel/Adam/Assign'CCN_1Conv_x0/convA11/kernel/Adam/read:024CCN_1Conv_x0/convA11/kernel/Adam/Initializer/zeros:0
?
$CCN_1Conv_x0/convA11/kernel/Adam_1:0)CCN_1Conv_x0/convA11/kernel/Adam_1/Assign)CCN_1Conv_x0/convA11/kernel/Adam_1/read:026CCN_1Conv_x0/convA11/kernel/Adam_1/Initializer/zeros:0
?
 CCN_1Conv_x0/convA11/bias/Adam:0%CCN_1Conv_x0/convA11/bias/Adam/Assign%CCN_1Conv_x0/convA11/bias/Adam/read:022CCN_1Conv_x0/convA11/bias/Adam/Initializer/zeros:0
?
"CCN_1Conv_x0/convA11/bias/Adam_1:0'CCN_1Conv_x0/convA11/bias/Adam_1/Assign'CCN_1Conv_x0/convA11/bias/Adam_1/read:024CCN_1Conv_x0/convA11/bias/Adam_1/Initializer/zeros:0
?
"CCN_1Conv_x0/convB11/kernel/Adam:0'CCN_1Conv_x0/convB11/kernel/Adam/Assign'CCN_1Conv_x0/convB11/kernel/Adam/read:024CCN_1Conv_x0/convB11/kernel/Adam/Initializer/zeros:0
?
$CCN_1Conv_x0/convB11/kernel/Adam_1:0)CCN_1Conv_x0/convB11/kernel/Adam_1/Assign)CCN_1Conv_x0/convB11/kernel/Adam_1/read:026CCN_1Conv_x0/convB11/kernel/Adam_1/Initializer/zeros:0
?
 CCN_1Conv_x0/convB11/bias/Adam:0%CCN_1Conv_x0/convB11/bias/Adam/Assign%CCN_1Conv_x0/convB11/bias/Adam/read:022CCN_1Conv_x0/convB11/bias/Adam/Initializer/zeros:0
?
"CCN_1Conv_x0/convB11/bias/Adam_1:0'CCN_1Conv_x0/convB11/bias/Adam_1/Assign'CCN_1Conv_x0/convB11/bias/Adam_1/read:024CCN_1Conv_x0/convB11/bias/Adam_1/Initializer/zeros:0
?
"CCN_1Conv_x0/convB21/kernel/Adam:0'CCN_1Conv_x0/convB21/kernel/Adam/Assign'CCN_1Conv_x0/convB21/kernel/Adam/read:024CCN_1Conv_x0/convB21/kernel/Adam/Initializer/zeros:0
?
$CCN_1Conv_x0/convB21/kernel/Adam_1:0)CCN_1Conv_x0/convB21/kernel/Adam_1/Assign)CCN_1Conv_x0/convB21/kernel/Adam_1/read:026CCN_1Conv_x0/convB21/kernel/Adam_1/Initializer/zeros:0
?
 CCN_1Conv_x0/convB21/bias/Adam:0%CCN_1Conv_x0/convB21/bias/Adam/Assign%CCN_1Conv_x0/convB21/bias/Adam/read:022CCN_1Conv_x0/convB21/bias/Adam/Initializer/zeros:0
?
"CCN_1Conv_x0/convB21/bias/Adam_1:0'CCN_1Conv_x0/convB21/bias/Adam_1/Assign'CCN_1Conv_x0/convB21/bias/Adam_1/read:024CCN_1Conv_x0/convB21/bias/Adam_1/Initializer/zeros:0
|
Conv_out__/beta/Adam:0Conv_out__/beta/Adam/AssignConv_out__/beta/Adam/read:02(Conv_out__/beta/Adam/Initializer/zeros:0
?
Conv_out__/beta/Adam_1:0Conv_out__/beta/Adam_1/AssignConv_out__/beta/Adam_1/read:02*Conv_out__/beta/Adam_1/Initializer/zeros:0
?
Conv_out__/gamma/Adam:0Conv_out__/gamma/Adam/AssignConv_out__/gamma/Adam/read:02)Conv_out__/gamma/Adam/Initializer/zeros:0
?
Conv_out__/gamma/Adam_1:0Conv_out__/gamma/Adam_1/AssignConv_out__/gamma/Adam_1/read:02+Conv_out__/gamma/Adam_1/Initializer/zeros:0
?
)Reconstruction_Output/dense/kernel/Adam:0.Reconstruction_Output/dense/kernel/Adam/Assign.Reconstruction_Output/dense/kernel/Adam/read:02;Reconstruction_Output/dense/kernel/Adam/Initializer/zeros:0
?
+Reconstruction_Output/dense/kernel/Adam_1:00Reconstruction_Output/dense/kernel/Adam_1/Assign0Reconstruction_Output/dense/kernel/Adam_1/read:02=Reconstruction_Output/dense/kernel/Adam_1/Initializer/zeros:0
?
'Reconstruction_Output/dense/bias/Adam:0,Reconstruction_Output/dense/bias/Adam/Assign,Reconstruction_Output/dense/bias/Adam/read:029Reconstruction_Output/dense/bias/Adam/Initializer/zeros:0
?
)Reconstruction_Output/dense/bias/Adam_1:0.Reconstruction_Output/dense/bias/Adam_1/Assign.Reconstruction_Output/dense/bias/Adam_1/read:02;Reconstruction_Output/dense/bias/Adam_1/Initializer/zeros:0
p
dense/kernel/Adam:0dense/kernel/Adam/Assigndense/kernel/Adam/read:02%dense/kernel/Adam/Initializer/zeros:0
x
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assigndense/kernel/Adam_1/read:02'dense/kernel/Adam_1/Initializer/zeros:0
h
dense/bias/Adam:0dense/bias/Adam/Assigndense/bias/Adam/read:02#dense/bias/Adam/Initializer/zeros:0
p
dense/bias/Adam_1:0dense/bias/Adam_1/Assigndense/bias/Adam_1/read:02%dense/bias/Adam_1/Initializer/zeros:0
?
%FCU_muiltDense_x0/dense/kernel/Adam:0*FCU_muiltDense_x0/dense/kernel/Adam/Assign*FCU_muiltDense_x0/dense/kernel/Adam/read:027FCU_muiltDense_x0/dense/kernel/Adam/Initializer/zeros:0
?
'FCU_muiltDense_x0/dense/kernel/Adam_1:0,FCU_muiltDense_x0/dense/kernel/Adam_1/Assign,FCU_muiltDense_x0/dense/kernel/Adam_1/read:029FCU_muiltDense_x0/dense/kernel/Adam_1/Initializer/zeros:0
?
#FCU_muiltDense_x0/dense/bias/Adam:0(FCU_muiltDense_x0/dense/bias/Adam/Assign(FCU_muiltDense_x0/dense/bias/Adam/read:025FCU_muiltDense_x0/dense/bias/Adam/Initializer/zeros:0
?
%FCU_muiltDense_x0/dense/bias/Adam_1:0*FCU_muiltDense_x0/dense/bias/Adam_1/Assign*FCU_muiltDense_x0/dense/bias/Adam_1/read:027FCU_muiltDense_x0/dense/bias/Adam_1/Initializer/zeros:0
?
FCU_muiltDense_x0/beta/Adam:0"FCU_muiltDense_x0/beta/Adam/Assign"FCU_muiltDense_x0/beta/Adam/read:02/FCU_muiltDense_x0/beta/Adam/Initializer/zeros:0
?
FCU_muiltDense_x0/beta/Adam_1:0$FCU_muiltDense_x0/beta/Adam_1/Assign$FCU_muiltDense_x0/beta/Adam_1/read:021FCU_muiltDense_x0/beta/Adam_1/Initializer/zeros:0
?
FCU_muiltDense_x0/gamma/Adam:0#FCU_muiltDense_x0/gamma/Adam/Assign#FCU_muiltDense_x0/gamma/Adam/read:020FCU_muiltDense_x0/gamma/Adam/Initializer/zeros:0
?
 FCU_muiltDense_x0/gamma/Adam_1:0%FCU_muiltDense_x0/gamma/Adam_1/Assign%FCU_muiltDense_x0/gamma/Adam_1/read:022FCU_muiltDense_x0/gamma/Adam_1/Initializer/zeros:0
?
Output_/dense/kernel/Adam:0 Output_/dense/kernel/Adam/Assign Output_/dense/kernel/Adam/read:02-Output_/dense/kernel/Adam/Initializer/zeros:0
?
Output_/dense/kernel/Adam_1:0"Output_/dense/kernel/Adam_1/Assign"Output_/dense/kernel/Adam_1/read:02/Output_/dense/kernel/Adam_1/Initializer/zeros:0
?
Output_/dense/bias/Adam:0Output_/dense/bias/Adam/AssignOutput_/dense/bias/Adam/read:02+Output_/dense/bias/Adam/Initializer/zeros:0
?
Output_/dense/bias/Adam_1:0 Output_/dense/bias/Adam_1/Assign Output_/dense/bias/Adam_1/read:02-Output_/dense/bias/Adam_1/Initializer/zeros:0
?
6CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage:0;CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/Assign;CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/read:02cond/Merge:0
?
4CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage:09CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/Assign9CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/read:02cond_1/Merge:0
?
6CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage:0;CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/Assign;CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/read:02cond_2/Merge:0
?
4CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage:09CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/Assign9CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/read:02cond_3/Merge:0
?
6CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage:0;CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/Assign;CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/read:02cond_4/Merge:0
?
4CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage:09CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/Assign9CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/read:02cond_5/Merge:0
?
6CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage:0;CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/Assign;CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/read:02cond_6/Merge:0
?
4CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage:09CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/Assign9CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/read:02cond_7/Merge:0
?
6CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage:0;CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/Assign;CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/read:02cond_8/Merge:0
?
4CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage:09CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/Assign9CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/read:02cond_9/Merge:0
?
6CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage:0;CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/Assign;CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/read:02cond_10/Merge:0
?
4CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage:09CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/Assign9CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/read:02cond_11/Merge:0
?
*Conv_out__/beta/ExponentialMovingAverage:0/Conv_out__/beta/ExponentialMovingAverage/Assign/Conv_out__/beta/ExponentialMovingAverage/read:02cond_12/Merge:0
?
+Conv_out__/gamma/ExponentialMovingAverage:00Conv_out__/gamma/ExponentialMovingAverage/Assign0Conv_out__/gamma/ExponentialMovingAverage/read:02cond_13/Merge:0
?
=Reconstruction_Output/dense/kernel/ExponentialMovingAverage:0BReconstruction_Output/dense/kernel/ExponentialMovingAverage/AssignBReconstruction_Output/dense/kernel/ExponentialMovingAverage/read:02cond_14/Merge:0
?
;Reconstruction_Output/dense/bias/ExponentialMovingAverage:0@Reconstruction_Output/dense/bias/ExponentialMovingAverage/Assign@Reconstruction_Output/dense/bias/ExponentialMovingAverage/read:02cond_15/Merge:0
?
'dense/kernel/ExponentialMovingAverage:0,dense/kernel/ExponentialMovingAverage/Assign,dense/kernel/ExponentialMovingAverage/read:02cond_16/Merge:0
?
%dense/bias/ExponentialMovingAverage:0*dense/bias/ExponentialMovingAverage/Assign*dense/bias/ExponentialMovingAverage/read:02cond_17/Merge:0
?
9FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage:0>FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/Assign>FCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/read:02cond_18/Merge:0
?
7FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage:0<FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/Assign<FCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/read:02cond_19/Merge:0
?
1FCU_muiltDense_x0/beta/ExponentialMovingAverage:06FCU_muiltDense_x0/beta/ExponentialMovingAverage/Assign6FCU_muiltDense_x0/beta/ExponentialMovingAverage/read:02cond_20/Merge:0
?
2FCU_muiltDense_x0/gamma/ExponentialMovingAverage:07FCU_muiltDense_x0/gamma/ExponentialMovingAverage/Assign7FCU_muiltDense_x0/gamma/ExponentialMovingAverage/read:02cond_21/Merge:0
?
/Output_/dense/kernel/ExponentialMovingAverage:04Output_/dense/kernel/ExponentialMovingAverage/Assign4Output_/dense/kernel/ExponentialMovingAverage/read:02cond_22/Merge:0
?
-Output_/dense/bias/ExponentialMovingAverage:02Output_/dense/bias/ExponentialMovingAverage/Assign2Output_/dense/bias/ExponentialMovingAverage/read:02cond_23/Merge:0
?
-BackupVariables/CCN_1Conv_x0/convA10/kernel:02BackupVariables/CCN_1Conv_x0/convA10/kernel/Assign2BackupVariables/CCN_1Conv_x0/convA10/kernel/read:02BackupVariables/cond/Merge:0
?
+BackupVariables/CCN_1Conv_x0/convA10/bias:00BackupVariables/CCN_1Conv_x0/convA10/bias/Assign0BackupVariables/CCN_1Conv_x0/convA10/bias/read:02BackupVariables/cond_1/Merge:0
?
-BackupVariables/CCN_1Conv_x0/convB10/kernel:02BackupVariables/CCN_1Conv_x0/convB10/kernel/Assign2BackupVariables/CCN_1Conv_x0/convB10/kernel/read:02BackupVariables/cond_2/Merge:0
?
+BackupVariables/CCN_1Conv_x0/convB10/bias:00BackupVariables/CCN_1Conv_x0/convB10/bias/Assign0BackupVariables/CCN_1Conv_x0/convB10/bias/read:02BackupVariables/cond_3/Merge:0
?
-BackupVariables/CCN_1Conv_x0/convB20/kernel:02BackupVariables/CCN_1Conv_x0/convB20/kernel/Assign2BackupVariables/CCN_1Conv_x0/convB20/kernel/read:02BackupVariables/cond_4/Merge:0
?
+BackupVariables/CCN_1Conv_x0/convB20/bias:00BackupVariables/CCN_1Conv_x0/convB20/bias/Assign0BackupVariables/CCN_1Conv_x0/convB20/bias/read:02BackupVariables/cond_5/Merge:0
?
-BackupVariables/CCN_1Conv_x0/convA11/kernel:02BackupVariables/CCN_1Conv_x0/convA11/kernel/Assign2BackupVariables/CCN_1Conv_x0/convA11/kernel/read:02BackupVariables/cond_6/Merge:0
?
+BackupVariables/CCN_1Conv_x0/convA11/bias:00BackupVariables/CCN_1Conv_x0/convA11/bias/Assign0BackupVariables/CCN_1Conv_x0/convA11/bias/read:02BackupVariables/cond_7/Merge:0
?
-BackupVariables/CCN_1Conv_x0/convB11/kernel:02BackupVariables/CCN_1Conv_x0/convB11/kernel/Assign2BackupVariables/CCN_1Conv_x0/convB11/kernel/read:02BackupVariables/cond_8/Merge:0
?
+BackupVariables/CCN_1Conv_x0/convB11/bias:00BackupVariables/CCN_1Conv_x0/convB11/bias/Assign0BackupVariables/CCN_1Conv_x0/convB11/bias/read:02BackupVariables/cond_9/Merge:0
?
-BackupVariables/CCN_1Conv_x0/convB21/kernel:02BackupVariables/CCN_1Conv_x0/convB21/kernel/Assign2BackupVariables/CCN_1Conv_x0/convB21/kernel/read:02BackupVariables/cond_10/Merge:0
?
+BackupVariables/CCN_1Conv_x0/convB21/bias:00BackupVariables/CCN_1Conv_x0/convB21/bias/Assign0BackupVariables/CCN_1Conv_x0/convB21/bias/read:02BackupVariables/cond_11/Merge:0
?
!BackupVariables/Conv_out__/beta:0&BackupVariables/Conv_out__/beta/Assign&BackupVariables/Conv_out__/beta/read:02BackupVariables/cond_12/Merge:0
?
"BackupVariables/Conv_out__/gamma:0'BackupVariables/Conv_out__/gamma/Assign'BackupVariables/Conv_out__/gamma/read:02BackupVariables/cond_13/Merge:0
?
4BackupVariables/Reconstruction_Output/dense/kernel:09BackupVariables/Reconstruction_Output/dense/kernel/Assign9BackupVariables/Reconstruction_Output/dense/kernel/read:02BackupVariables/cond_14/Merge:0
?
2BackupVariables/Reconstruction_Output/dense/bias:07BackupVariables/Reconstruction_Output/dense/bias/Assign7BackupVariables/Reconstruction_Output/dense/bias/read:02BackupVariables/cond_15/Merge:0
?
BackupVariables/dense/kernel:0#BackupVariables/dense/kernel/Assign#BackupVariables/dense/kernel/read:02BackupVariables/cond_16/Merge:0
?
BackupVariables/dense/bias:0!BackupVariables/dense/bias/Assign!BackupVariables/dense/bias/read:02BackupVariables/cond_17/Merge:0
?
0BackupVariables/FCU_muiltDense_x0/dense/kernel:05BackupVariables/FCU_muiltDense_x0/dense/kernel/Assign5BackupVariables/FCU_muiltDense_x0/dense/kernel/read:02BackupVariables/cond_18/Merge:0
?
.BackupVariables/FCU_muiltDense_x0/dense/bias:03BackupVariables/FCU_muiltDense_x0/dense/bias/Assign3BackupVariables/FCU_muiltDense_x0/dense/bias/read:02BackupVariables/cond_19/Merge:0
?
(BackupVariables/FCU_muiltDense_x0/beta:0-BackupVariables/FCU_muiltDense_x0/beta/Assign-BackupVariables/FCU_muiltDense_x0/beta/read:02BackupVariables/cond_20/Merge:0
?
)BackupVariables/FCU_muiltDense_x0/gamma:0.BackupVariables/FCU_muiltDense_x0/gamma/Assign.BackupVariables/FCU_muiltDense_x0/gamma/read:02BackupVariables/cond_21/Merge:0
?
&BackupVariables/Output_/dense/kernel:0+BackupVariables/Output_/dense/kernel/Assign+BackupVariables/Output_/dense/kernel/read:02BackupVariables/cond_22/Merge:0
?
$BackupVariables/Output_/dense/bias:0)BackupVariables/Output_/dense/bias/Assign)BackupVariables/Output_/dense/bias/read:02BackupVariables/cond_23/Merge:0"
train_op

Adam"?
trainable_variables??
?
CCN_1Conv_x0/convA10/kernel:0"CCN_1Conv_x0/convA10/kernel/Assign"CCN_1Conv_x0/convA10/kernel/read:028CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convA10/bias:0 CCN_1Conv_x0/convA10/bias/Assign CCN_1Conv_x0/convA10/bias/read:02-CCN_1Conv_x0/convA10/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB10/kernel:0"CCN_1Conv_x0/convB10/kernel/Assign"CCN_1Conv_x0/convB10/kernel/read:028CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB10/bias:0 CCN_1Conv_x0/convB10/bias/Assign CCN_1Conv_x0/convB10/bias/read:02-CCN_1Conv_x0/convB10/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB20/kernel:0"CCN_1Conv_x0/convB20/kernel/Assign"CCN_1Conv_x0/convB20/kernel/read:028CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB20/bias:0 CCN_1Conv_x0/convB20/bias/Assign CCN_1Conv_x0/convB20/bias/read:02-CCN_1Conv_x0/convB20/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convA11/kernel:0"CCN_1Conv_x0/convA11/kernel/Assign"CCN_1Conv_x0/convA11/kernel/read:028CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convA11/bias:0 CCN_1Conv_x0/convA11/bias/Assign CCN_1Conv_x0/convA11/bias/read:02-CCN_1Conv_x0/convA11/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB11/kernel:0"CCN_1Conv_x0/convB11/kernel/Assign"CCN_1Conv_x0/convB11/kernel/read:028CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB11/bias:0 CCN_1Conv_x0/convB11/bias/Assign CCN_1Conv_x0/convB11/bias/read:02-CCN_1Conv_x0/convB11/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB21/kernel:0"CCN_1Conv_x0/convB21/kernel/Assign"CCN_1Conv_x0/convB21/kernel/read:028CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB21/bias:0 CCN_1Conv_x0/convB21/bias/Assign CCN_1Conv_x0/convB21/bias/read:02-CCN_1Conv_x0/convB21/bias/Initializer/zeros:08
j
Conv_out__/beta:0Conv_out__/beta/AssignConv_out__/beta/read:02#Conv_out__/beta/Initializer/zeros:08
m
Conv_out__/gamma:0Conv_out__/gamma/AssignConv_out__/gamma/read:02#Conv_out__/gamma/Initializer/ones:08
?
$Reconstruction_Output/dense/kernel:0)Reconstruction_Output/dense/kernel/Assign)Reconstruction_Output/dense/kernel/read:02?Reconstruction_Output/dense/kernel/Initializer/random_uniform:08
?
"Reconstruction_Output/dense/bias:0'Reconstruction_Output/dense/bias/Assign'Reconstruction_Output/dense/bias/read:024Reconstruction_Output/dense/bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
?
 FCU_muiltDense_x0/dense/kernel:0%FCU_muiltDense_x0/dense/kernel/Assign%FCU_muiltDense_x0/dense/kernel/read:02;FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform:08
?
FCU_muiltDense_x0/dense/bias:0#FCU_muiltDense_x0/dense/bias/Assign#FCU_muiltDense_x0/dense/bias/read:020FCU_muiltDense_x0/dense/bias/Initializer/zeros:08
?
FCU_muiltDense_x0/beta:0FCU_muiltDense_x0/beta/AssignFCU_muiltDense_x0/beta/read:02*FCU_muiltDense_x0/beta/Initializer/zeros:08
?
FCU_muiltDense_x0/gamma:0FCU_muiltDense_x0/gamma/AssignFCU_muiltDense_x0/gamma/read:02*FCU_muiltDense_x0/gamma/Initializer/ones:08
?
Output_/dense/kernel:0Output_/dense/kernel/AssignOutput_/dense/kernel/read:021Output_/dense/kernel/Initializer/random_uniform:08
v
Output_/dense/bias:0Output_/dense/bias/AssignOutput_/dense/bias/read:02&Output_/dense/bias/Initializer/zeros:08"?
moving_average_variables??
?
CCN_1Conv_x0/convA10/kernel:0"CCN_1Conv_x0/convA10/kernel/Assign"CCN_1Conv_x0/convA10/kernel/read:028CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convA10/bias:0 CCN_1Conv_x0/convA10/bias/Assign CCN_1Conv_x0/convA10/bias/read:02-CCN_1Conv_x0/convA10/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB10/kernel:0"CCN_1Conv_x0/convB10/kernel/Assign"CCN_1Conv_x0/convB10/kernel/read:028CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB10/bias:0 CCN_1Conv_x0/convB10/bias/Assign CCN_1Conv_x0/convB10/bias/read:02-CCN_1Conv_x0/convB10/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB20/kernel:0"CCN_1Conv_x0/convB20/kernel/Assign"CCN_1Conv_x0/convB20/kernel/read:028CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB20/bias:0 CCN_1Conv_x0/convB20/bias/Assign CCN_1Conv_x0/convB20/bias/read:02-CCN_1Conv_x0/convB20/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convA11/kernel:0"CCN_1Conv_x0/convA11/kernel/Assign"CCN_1Conv_x0/convA11/kernel/read:028CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convA11/bias:0 CCN_1Conv_x0/convA11/bias/Assign CCN_1Conv_x0/convA11/bias/read:02-CCN_1Conv_x0/convA11/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB11/kernel:0"CCN_1Conv_x0/convB11/kernel/Assign"CCN_1Conv_x0/convB11/kernel/read:028CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB11/bias:0 CCN_1Conv_x0/convB11/bias/Assign CCN_1Conv_x0/convB11/bias/read:02-CCN_1Conv_x0/convB11/bias/Initializer/zeros:08
?
CCN_1Conv_x0/convB21/kernel:0"CCN_1Conv_x0/convB21/kernel/Assign"CCN_1Conv_x0/convB21/kernel/read:028CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform:08
?
CCN_1Conv_x0/convB21/bias:0 CCN_1Conv_x0/convB21/bias/Assign CCN_1Conv_x0/convB21/bias/read:02-CCN_1Conv_x0/convB21/bias/Initializer/zeros:08
j
Conv_out__/beta:0Conv_out__/beta/AssignConv_out__/beta/read:02#Conv_out__/beta/Initializer/zeros:08
m
Conv_out__/gamma:0Conv_out__/gamma/AssignConv_out__/gamma/read:02#Conv_out__/gamma/Initializer/ones:08
?
$Reconstruction_Output/dense/kernel:0)Reconstruction_Output/dense/kernel/Assign)Reconstruction_Output/dense/kernel/read:02?Reconstruction_Output/dense/kernel/Initializer/random_uniform:08
?
"Reconstruction_Output/dense/bias:0'Reconstruction_Output/dense/bias/Assign'Reconstruction_Output/dense/bias/read:024Reconstruction_Output/dense/bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
?
 FCU_muiltDense_x0/dense/kernel:0%FCU_muiltDense_x0/dense/kernel/Assign%FCU_muiltDense_x0/dense/kernel/read:02;FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform:08
?
FCU_muiltDense_x0/dense/bias:0#FCU_muiltDense_x0/dense/bias/Assign#FCU_muiltDense_x0/dense/bias/read:020FCU_muiltDense_x0/dense/bias/Initializer/zeros:08
?
FCU_muiltDense_x0/beta:0FCU_muiltDense_x0/beta/AssignFCU_muiltDense_x0/beta/read:02*FCU_muiltDense_x0/beta/Initializer/zeros:08
?
FCU_muiltDense_x0/gamma:0FCU_muiltDense_x0/gamma/AssignFCU_muiltDense_x0/gamma/read:02*FCU_muiltDense_x0/gamma/Initializer/ones:08
?
Output_/dense/kernel:0Output_/dense/kernel/AssignOutput_/dense/kernel/read:021Output_/dense/kernel/Initializer/random_uniform:08
v
Output_/dense/bias:0Output_/dense/bias/AssignOutput_/dense/bias/read:02&Output_/dense/bias/Initializer/zeros:08"
	iterators

IteratorV2:0"?
model_variables??
j
Conv_out__/beta:0Conv_out__/beta/AssignConv_out__/beta/read:02#Conv_out__/beta/Initializer/zeros:08
m
Conv_out__/gamma:0Conv_out__/gamma/AssignConv_out__/gamma/read:02#Conv_out__/gamma/Initializer/ones:08
?
FCU_muiltDense_x0/beta:0FCU_muiltDense_x0/beta/AssignFCU_muiltDense_x0/beta/read:02*FCU_muiltDense_x0/beta/Initializer/zeros:08
?
FCU_muiltDense_x0/gamma:0FCU_muiltDense_x0/gamma/AssignFCU_muiltDense_x0/gamma/read:02*FCU_muiltDense_x0/gamma/Initializer/ones:08"?
regularization_losses?
?
6My_GPU_1/CCN_1Conv_x0/convA10/kernel/Regularizer/add:0
6My_GPU_1/CCN_1Conv_x0/convB10/kernel/Regularizer/add:0
6My_GPU_1/CCN_1Conv_x0/convB20/kernel/Regularizer/add:0
6My_GPU_1/CCN_1Conv_x0/convA11/kernel/Regularizer/add:0
6My_GPU_1/CCN_1Conv_x0/convB11/kernel/Regularizer/add:0
6My_GPU_1/CCN_1Conv_x0/convB21/kernel/Regularizer/add:0
=My_GPU_1/Reconstruction_Output/dense/kernel/Regularizer/add:0
'My_GPU_1/dense/kernel/Regularizer/add:0
9My_GPU_1/FCU_muiltDense_x0/dense/kernel/Regularizer/add:0
/My_GPU_1/Output_/dense/kernel/Regularizer/add:0"?
	summaries?
?
My_GPU_1/Total_Loss:0
My_GPU_1/Main_loss_value:0
My_GPU_1/Loss_l2_loss:0
CCN_1Conv_x0/convA10/kernel_1:0
CCN_1Conv_x0/convA10/bias_1:0
CCN_1Conv_x0/convB10/kernel_1:0
CCN_1Conv_x0/convB10/bias_1:0
CCN_1Conv_x0/convB20/kernel_1:0
CCN_1Conv_x0/convB20/bias_1:0
CCN_1Conv_x0/convA11/kernel_1:0
CCN_1Conv_x0/convA11/bias_1:0
CCN_1Conv_x0/convB11/kernel_1:0
CCN_1Conv_x0/convB11/bias_1:0
CCN_1Conv_x0/convB21/kernel_1:0
CCN_1Conv_x0/convB21/bias_1:0
Conv_out__/beta_1:0
Conv_out__/gamma_1:0
&Reconstruction_Output/dense/kernel_1:0
$Reconstruction_Output/dense/bias_1:0
dense/kernel_1:0
dense/bias_1:0
"FCU_muiltDense_x0/dense/kernel_1:0
 FCU_muiltDense_x0/dense/bias_1:0
FCU_muiltDense_x0/beta_1:0
FCU_muiltDense_x0/gamma_1:0
Output_/dense/kernel_1:0
Output_/dense/bias_1:0
Result_items/Loss:0
Result_items/Accuracy:0
"Result_items/Reconstruction_Loss:0
Result_items/Custom_Loss:0"??
cond_context????
?
cond/cond_textcond/pred_id:0cond/switch_t:0 *?
CCN_1Conv_x0/convA10/kernel:0
cond/pred_id:0
cond/read/Switch:1
cond/read:0
cond/switch_t:0 
cond/pred_id:0cond/pred_id:03
CCN_1Conv_x0/convA10/kernel:0cond/read/Switch:1
?
cond/cond_text_1cond/pred_id:0cond/switch_f:0*?
8CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform:0
cond/Switch_1:0
cond/Switch_1:1
cond/pred_id:0
cond/switch_f:0 
cond/pred_id:0cond/pred_id:0K
8CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform:0cond/Switch_1:0
?
CCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/cond_textCCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/switch_t:0 *?
CCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/pred_id:0
GCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/read/Switch:1
@CCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/read:0
DCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convA10/kernel:0?
CCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/pred_id:0h
CCN_1Conv_x0/convA10/kernel:0GCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/read/Switch:1
?
ECCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/cond_text_1CCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/switch_f:0*?
DCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/Switch_1:0
DCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/Switch_1:1
CCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/pred_id:0
DCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/switch_f:0
8CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform:0?
8CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform:0DCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/Switch_1:0?
CCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convA10/kernel/ExponentialMovingAverage/cond/pred_id:0
?
cond_1/cond_textcond_1/pred_id:0cond_1/switch_t:0 *?
CCN_1Conv_x0/convA10/bias:0
cond_1/pred_id:0
cond_1/read/Switch:1
cond_1/read:0
cond_1/switch_t:0$
cond_1/pred_id:0cond_1/pred_id:03
CCN_1Conv_x0/convA10/bias:0cond_1/read/Switch:1
?
cond_1/cond_text_1cond_1/pred_id:0cond_1/switch_f:0*?
-CCN_1Conv_x0/convA10/bias/Initializer/zeros:0
cond_1/Switch_1:0
cond_1/Switch_1:1
cond_1/pred_id:0
cond_1/switch_f:0B
-CCN_1Conv_x0/convA10/bias/Initializer/zeros:0cond_1/Switch_1:0$
cond_1/pred_id:0cond_1/pred_id:0
?
ACCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/cond_textACCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/switch_t:0 *?
ACCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/pred_id:0
ECCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/read/Switch:1
>CCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/read:0
BCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convA10/bias:0?
ACCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/pred_id:0d
CCN_1Conv_x0/convA10/bias:0ECCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/read/Switch:1
?
CCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/cond_text_1ACCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/switch_f:0*?
BCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/Switch_1:0
BCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/Switch_1:1
ACCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/pred_id:0
BCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/switch_f:0
-CCN_1Conv_x0/convA10/bias/Initializer/zeros:0s
-CCN_1Conv_x0/convA10/bias/Initializer/zeros:0BCCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/Switch_1:0?
ACCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convA10/bias/ExponentialMovingAverage/cond/pred_id:0
?
cond_2/cond_textcond_2/pred_id:0cond_2/switch_t:0 *?
CCN_1Conv_x0/convB10/kernel:0
cond_2/pred_id:0
cond_2/read/Switch:1
cond_2/read:0
cond_2/switch_t:0$
cond_2/pred_id:0cond_2/pred_id:05
CCN_1Conv_x0/convB10/kernel:0cond_2/read/Switch:1
?
cond_2/cond_text_1cond_2/pred_id:0cond_2/switch_f:0*?
8CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform:0
cond_2/Switch_1:0
cond_2/Switch_1:1
cond_2/pred_id:0
cond_2/switch_f:0$
cond_2/pred_id:0cond_2/pred_id:0M
8CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform:0cond_2/Switch_1:0
?
CCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/cond_textCCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/switch_t:0 *?
CCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/pred_id:0
GCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/read/Switch:1
@CCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/read:0
DCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convB10/kernel:0?
CCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/pred_id:0h
CCN_1Conv_x0/convB10/kernel:0GCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/read/Switch:1
?
ECCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/cond_text_1CCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/switch_f:0*?
DCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/Switch_1:0
DCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/Switch_1:1
CCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/pred_id:0
DCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/switch_f:0
8CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform:0?
CCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/pred_id:0?
8CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform:0DCCN_1Conv_x0/convB10/kernel/ExponentialMovingAverage/cond/Switch_1:0
?
cond_3/cond_textcond_3/pred_id:0cond_3/switch_t:0 *?
CCN_1Conv_x0/convB10/bias:0
cond_3/pred_id:0
cond_3/read/Switch:1
cond_3/read:0
cond_3/switch_t:0$
cond_3/pred_id:0cond_3/pred_id:03
CCN_1Conv_x0/convB10/bias:0cond_3/read/Switch:1
?
cond_3/cond_text_1cond_3/pred_id:0cond_3/switch_f:0*?
-CCN_1Conv_x0/convB10/bias/Initializer/zeros:0
cond_3/Switch_1:0
cond_3/Switch_1:1
cond_3/pred_id:0
cond_3/switch_f:0$
cond_3/pred_id:0cond_3/pred_id:0B
-CCN_1Conv_x0/convB10/bias/Initializer/zeros:0cond_3/Switch_1:0
?
ACCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/cond_textACCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/switch_t:0 *?
ACCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/pred_id:0
ECCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/read/Switch:1
>CCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/read:0
BCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convB10/bias:0d
CCN_1Conv_x0/convB10/bias:0ECCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/read/Switch:1?
ACCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/pred_id:0
?
CCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/cond_text_1ACCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/switch_f:0*?
BCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/Switch_1:0
BCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/Switch_1:1
ACCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/pred_id:0
BCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/switch_f:0
-CCN_1Conv_x0/convB10/bias/Initializer/zeros:0?
ACCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/pred_id:0s
-CCN_1Conv_x0/convB10/bias/Initializer/zeros:0BCCN_1Conv_x0/convB10/bias/ExponentialMovingAverage/cond/Switch_1:0
?
cond_4/cond_textcond_4/pred_id:0cond_4/switch_t:0 *?
CCN_1Conv_x0/convB20/kernel:0
cond_4/pred_id:0
cond_4/read/Switch:1
cond_4/read:0
cond_4/switch_t:0$
cond_4/pred_id:0cond_4/pred_id:05
CCN_1Conv_x0/convB20/kernel:0cond_4/read/Switch:1
?
cond_4/cond_text_1cond_4/pred_id:0cond_4/switch_f:0*?
8CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform:0
cond_4/Switch_1:0
cond_4/Switch_1:1
cond_4/pred_id:0
cond_4/switch_f:0$
cond_4/pred_id:0cond_4/pred_id:0M
8CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform:0cond_4/Switch_1:0
?
CCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/cond_textCCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/switch_t:0 *?
CCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/pred_id:0
GCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/read/Switch:1
@CCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/read:0
DCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convB20/kernel:0?
CCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/pred_id:0h
CCN_1Conv_x0/convB20/kernel:0GCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/read/Switch:1
?
ECCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/cond_text_1CCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/switch_f:0*?
DCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/Switch_1:0
DCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/Switch_1:1
CCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/pred_id:0
DCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/switch_f:0
8CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform:0?
8CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform:0DCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/Switch_1:0?
CCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convB20/kernel/ExponentialMovingAverage/cond/pred_id:0
?
cond_5/cond_textcond_5/pred_id:0cond_5/switch_t:0 *?
CCN_1Conv_x0/convB20/bias:0
cond_5/pred_id:0
cond_5/read/Switch:1
cond_5/read:0
cond_5/switch_t:0$
cond_5/pred_id:0cond_5/pred_id:03
CCN_1Conv_x0/convB20/bias:0cond_5/read/Switch:1
?
cond_5/cond_text_1cond_5/pred_id:0cond_5/switch_f:0*?
-CCN_1Conv_x0/convB20/bias/Initializer/zeros:0
cond_5/Switch_1:0
cond_5/Switch_1:1
cond_5/pred_id:0
cond_5/switch_f:0$
cond_5/pred_id:0cond_5/pred_id:0B
-CCN_1Conv_x0/convB20/bias/Initializer/zeros:0cond_5/Switch_1:0
?
ACCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/cond_textACCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/switch_t:0 *?
ACCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/pred_id:0
ECCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/read/Switch:1
>CCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/read:0
BCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convB20/bias:0?
ACCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/pred_id:0d
CCN_1Conv_x0/convB20/bias:0ECCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/read/Switch:1
?
CCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/cond_text_1ACCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/switch_f:0*?
BCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/Switch_1:0
BCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/Switch_1:1
ACCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/pred_id:0
BCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/switch_f:0
-CCN_1Conv_x0/convB20/bias/Initializer/zeros:0?
ACCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/pred_id:0s
-CCN_1Conv_x0/convB20/bias/Initializer/zeros:0BCCN_1Conv_x0/convB20/bias/ExponentialMovingAverage/cond/Switch_1:0
?
cond_6/cond_textcond_6/pred_id:0cond_6/switch_t:0 *?
CCN_1Conv_x0/convA11/kernel:0
cond_6/pred_id:0
cond_6/read/Switch:1
cond_6/read:0
cond_6/switch_t:05
CCN_1Conv_x0/convA11/kernel:0cond_6/read/Switch:1$
cond_6/pred_id:0cond_6/pred_id:0
?
cond_6/cond_text_1cond_6/pred_id:0cond_6/switch_f:0*?
8CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform:0
cond_6/Switch_1:0
cond_6/Switch_1:1
cond_6/pred_id:0
cond_6/switch_f:0$
cond_6/pred_id:0cond_6/pred_id:0M
8CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform:0cond_6/Switch_1:0
?
CCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/cond_textCCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/switch_t:0 *?
CCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/pred_id:0
GCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/read/Switch:1
@CCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/read:0
DCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convA11/kernel:0?
CCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/pred_id:0h
CCN_1Conv_x0/convA11/kernel:0GCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/read/Switch:1
?
ECCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/cond_text_1CCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/switch_f:0*?
DCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/Switch_1:0
DCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/Switch_1:1
CCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/pred_id:0
DCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/switch_f:0
8CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform:0?
CCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/pred_id:0?
8CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform:0DCCN_1Conv_x0/convA11/kernel/ExponentialMovingAverage/cond/Switch_1:0
?
cond_7/cond_textcond_7/pred_id:0cond_7/switch_t:0 *?
CCN_1Conv_x0/convA11/bias:0
cond_7/pred_id:0
cond_7/read/Switch:1
cond_7/read:0
cond_7/switch_t:0$
cond_7/pred_id:0cond_7/pred_id:03
CCN_1Conv_x0/convA11/bias:0cond_7/read/Switch:1
?
cond_7/cond_text_1cond_7/pred_id:0cond_7/switch_f:0*?
-CCN_1Conv_x0/convA11/bias/Initializer/zeros:0
cond_7/Switch_1:0
cond_7/Switch_1:1
cond_7/pred_id:0
cond_7/switch_f:0B
-CCN_1Conv_x0/convA11/bias/Initializer/zeros:0cond_7/Switch_1:0$
cond_7/pred_id:0cond_7/pred_id:0
?
ACCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/cond_textACCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/switch_t:0 *?
ACCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/pred_id:0
ECCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/read/Switch:1
>CCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/read:0
BCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convA11/bias:0?
ACCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/pred_id:0d
CCN_1Conv_x0/convA11/bias:0ECCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/read/Switch:1
?
CCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/cond_text_1ACCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/switch_f:0*?
BCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/Switch_1:0
BCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/Switch_1:1
ACCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/pred_id:0
BCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/switch_f:0
-CCN_1Conv_x0/convA11/bias/Initializer/zeros:0s
-CCN_1Conv_x0/convA11/bias/Initializer/zeros:0BCCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/Switch_1:0?
ACCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convA11/bias/ExponentialMovingAverage/cond/pred_id:0
?
cond_8/cond_textcond_8/pred_id:0cond_8/switch_t:0 *?
CCN_1Conv_x0/convB11/kernel:0
cond_8/pred_id:0
cond_8/read/Switch:1
cond_8/read:0
cond_8/switch_t:05
CCN_1Conv_x0/convB11/kernel:0cond_8/read/Switch:1$
cond_8/pred_id:0cond_8/pred_id:0
?
cond_8/cond_text_1cond_8/pred_id:0cond_8/switch_f:0*?
8CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform:0
cond_8/Switch_1:0
cond_8/Switch_1:1
cond_8/pred_id:0
cond_8/switch_f:0$
cond_8/pred_id:0cond_8/pred_id:0M
8CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform:0cond_8/Switch_1:0
?
CCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/cond_textCCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/switch_t:0 *?
CCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/pred_id:0
GCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/read/Switch:1
@CCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/read:0
DCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convB11/kernel:0h
CCN_1Conv_x0/convB11/kernel:0GCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/read/Switch:1?
CCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/pred_id:0
?
ECCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/cond_text_1CCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/switch_f:0*?
DCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/Switch_1:0
DCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/Switch_1:1
CCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/pred_id:0
DCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/switch_f:0
8CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform:0?
CCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/pred_id:0?
8CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform:0DCCN_1Conv_x0/convB11/kernel/ExponentialMovingAverage/cond/Switch_1:0
?
cond_9/cond_textcond_9/pred_id:0cond_9/switch_t:0 *?
CCN_1Conv_x0/convB11/bias:0
cond_9/pred_id:0
cond_9/read/Switch:1
cond_9/read:0
cond_9/switch_t:03
CCN_1Conv_x0/convB11/bias:0cond_9/read/Switch:1$
cond_9/pred_id:0cond_9/pred_id:0
?
cond_9/cond_text_1cond_9/pred_id:0cond_9/switch_f:0*?
-CCN_1Conv_x0/convB11/bias/Initializer/zeros:0
cond_9/Switch_1:0
cond_9/Switch_1:1
cond_9/pred_id:0
cond_9/switch_f:0$
cond_9/pred_id:0cond_9/pred_id:0B
-CCN_1Conv_x0/convB11/bias/Initializer/zeros:0cond_9/Switch_1:0
?
ACCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/cond_textACCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/switch_t:0 *?
ACCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/pred_id:0
ECCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/read/Switch:1
>CCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/read:0
BCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convB11/bias:0?
ACCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/pred_id:0d
CCN_1Conv_x0/convB11/bias:0ECCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/read/Switch:1
?
CCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/cond_text_1ACCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/switch_f:0*?
BCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/Switch_1:0
BCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/Switch_1:1
ACCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/pred_id:0
BCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/switch_f:0
-CCN_1Conv_x0/convB11/bias/Initializer/zeros:0?
ACCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/pred_id:0s
-CCN_1Conv_x0/convB11/bias/Initializer/zeros:0BCCN_1Conv_x0/convB11/bias/ExponentialMovingAverage/cond/Switch_1:0
?
cond_10/cond_textcond_10/pred_id:0cond_10/switch_t:0 *?
CCN_1Conv_x0/convB21/kernel:0
cond_10/pred_id:0
cond_10/read/Switch:1
cond_10/read:0
cond_10/switch_t:0&
cond_10/pred_id:0cond_10/pred_id:06
CCN_1Conv_x0/convB21/kernel:0cond_10/read/Switch:1
?
cond_10/cond_text_1cond_10/pred_id:0cond_10/switch_f:0*?
8CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform:0
cond_10/Switch_1:0
cond_10/Switch_1:1
cond_10/pred_id:0
cond_10/switch_f:0N
8CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform:0cond_10/Switch_1:0&
cond_10/pred_id:0cond_10/pred_id:0
?
CCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/cond_textCCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/switch_t:0 *?
CCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/pred_id:0
GCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/read/Switch:1
@CCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/read:0
DCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convB21/kernel:0?
CCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/pred_id:0h
CCN_1Conv_x0/convB21/kernel:0GCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/read/Switch:1
?
ECCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/cond_text_1CCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/pred_id:0DCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/switch_f:0*?
DCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/Switch_1:0
DCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/Switch_1:1
CCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/pred_id:0
DCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/switch_f:0
8CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform:0?
8CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform:0DCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/Switch_1:0?
CCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/pred_id:0CCCN_1Conv_x0/convB21/kernel/ExponentialMovingAverage/cond/pred_id:0
?
cond_11/cond_textcond_11/pred_id:0cond_11/switch_t:0 *?
CCN_1Conv_x0/convB21/bias:0
cond_11/pred_id:0
cond_11/read/Switch:1
cond_11/read:0
cond_11/switch_t:0&
cond_11/pred_id:0cond_11/pred_id:04
CCN_1Conv_x0/convB21/bias:0cond_11/read/Switch:1
?
cond_11/cond_text_1cond_11/pred_id:0cond_11/switch_f:0*?
-CCN_1Conv_x0/convB21/bias/Initializer/zeros:0
cond_11/Switch_1:0
cond_11/Switch_1:1
cond_11/pred_id:0
cond_11/switch_f:0C
-CCN_1Conv_x0/convB21/bias/Initializer/zeros:0cond_11/Switch_1:0&
cond_11/pred_id:0cond_11/pred_id:0
?
ACCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/cond_textACCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/switch_t:0 *?
ACCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/pred_id:0
ECCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/read/Switch:1
>CCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/read:0
BCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/switch_t:0
CCN_1Conv_x0/convB21/bias:0?
ACCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/pred_id:0d
CCN_1Conv_x0/convB21/bias:0ECCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/read/Switch:1
?
CCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/cond_text_1ACCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/pred_id:0BCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/switch_f:0*?
BCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/Switch_1:0
BCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/Switch_1:1
ACCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/pred_id:0
BCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/switch_f:0
-CCN_1Conv_x0/convB21/bias/Initializer/zeros:0s
-CCN_1Conv_x0/convB21/bias/Initializer/zeros:0BCCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/Switch_1:0?
ACCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/pred_id:0ACCN_1Conv_x0/convB21/bias/ExponentialMovingAverage/cond/pred_id:0
?
cond_12/cond_textcond_12/pred_id:0cond_12/switch_t:0 *?
Conv_out__/beta:0
cond_12/pred_id:0
cond_12/read/Switch:1
cond_12/read:0
cond_12/switch_t:0&
cond_12/pred_id:0cond_12/pred_id:0*
Conv_out__/beta:0cond_12/read/Switch:1
?
cond_12/cond_text_1cond_12/pred_id:0cond_12/switch_f:0*?
#Conv_out__/beta/Initializer/zeros:0
cond_12/Switch_1:0
cond_12/Switch_1:1
cond_12/pred_id:0
cond_12/switch_f:0&
cond_12/pred_id:0cond_12/pred_id:09
#Conv_out__/beta/Initializer/zeros:0cond_12/Switch_1:0
?
7Conv_out__/beta/ExponentialMovingAverage/cond/cond_text7Conv_out__/beta/ExponentialMovingAverage/cond/pred_id:08Conv_out__/beta/ExponentialMovingAverage/cond/switch_t:0 *?
7Conv_out__/beta/ExponentialMovingAverage/cond/pred_id:0
;Conv_out__/beta/ExponentialMovingAverage/cond/read/Switch:1
4Conv_out__/beta/ExponentialMovingAverage/cond/read:0
8Conv_out__/beta/ExponentialMovingAverage/cond/switch_t:0
Conv_out__/beta:0P
Conv_out__/beta:0;Conv_out__/beta/ExponentialMovingAverage/cond/read/Switch:1r
7Conv_out__/beta/ExponentialMovingAverage/cond/pred_id:07Conv_out__/beta/ExponentialMovingAverage/cond/pred_id:0
?
9Conv_out__/beta/ExponentialMovingAverage/cond/cond_text_17Conv_out__/beta/ExponentialMovingAverage/cond/pred_id:08Conv_out__/beta/ExponentialMovingAverage/cond/switch_f:0*?
8Conv_out__/beta/ExponentialMovingAverage/cond/Switch_1:0
8Conv_out__/beta/ExponentialMovingAverage/cond/Switch_1:1
7Conv_out__/beta/ExponentialMovingAverage/cond/pred_id:0
8Conv_out__/beta/ExponentialMovingAverage/cond/switch_f:0
#Conv_out__/beta/Initializer/zeros:0r
7Conv_out__/beta/ExponentialMovingAverage/cond/pred_id:07Conv_out__/beta/ExponentialMovingAverage/cond/pred_id:0_
#Conv_out__/beta/Initializer/zeros:08Conv_out__/beta/ExponentialMovingAverage/cond/Switch_1:0
?
cond_13/cond_textcond_13/pred_id:0cond_13/switch_t:0 *?
Conv_out__/gamma:0
cond_13/pred_id:0
cond_13/read/Switch:1
cond_13/read:0
cond_13/switch_t:0&
cond_13/pred_id:0cond_13/pred_id:0+
Conv_out__/gamma:0cond_13/read/Switch:1
?
cond_13/cond_text_1cond_13/pred_id:0cond_13/switch_f:0*?
#Conv_out__/gamma/Initializer/ones:0
cond_13/Switch_1:0
cond_13/Switch_1:1
cond_13/pred_id:0
cond_13/switch_f:0&
cond_13/pred_id:0cond_13/pred_id:09
#Conv_out__/gamma/Initializer/ones:0cond_13/Switch_1:0
?
8Conv_out__/gamma/ExponentialMovingAverage/cond/cond_text8Conv_out__/gamma/ExponentialMovingAverage/cond/pred_id:09Conv_out__/gamma/ExponentialMovingAverage/cond/switch_t:0 *?
8Conv_out__/gamma/ExponentialMovingAverage/cond/pred_id:0
<Conv_out__/gamma/ExponentialMovingAverage/cond/read/Switch:1
5Conv_out__/gamma/ExponentialMovingAverage/cond/read:0
9Conv_out__/gamma/ExponentialMovingAverage/cond/switch_t:0
Conv_out__/gamma:0R
Conv_out__/gamma:0<Conv_out__/gamma/ExponentialMovingAverage/cond/read/Switch:1t
8Conv_out__/gamma/ExponentialMovingAverage/cond/pred_id:08Conv_out__/gamma/ExponentialMovingAverage/cond/pred_id:0
?
:Conv_out__/gamma/ExponentialMovingAverage/cond/cond_text_18Conv_out__/gamma/ExponentialMovingAverage/cond/pred_id:09Conv_out__/gamma/ExponentialMovingAverage/cond/switch_f:0*?
9Conv_out__/gamma/ExponentialMovingAverage/cond/Switch_1:0
9Conv_out__/gamma/ExponentialMovingAverage/cond/Switch_1:1
8Conv_out__/gamma/ExponentialMovingAverage/cond/pred_id:0
9Conv_out__/gamma/ExponentialMovingAverage/cond/switch_f:0
#Conv_out__/gamma/Initializer/ones:0t
8Conv_out__/gamma/ExponentialMovingAverage/cond/pred_id:08Conv_out__/gamma/ExponentialMovingAverage/cond/pred_id:0`
#Conv_out__/gamma/Initializer/ones:09Conv_out__/gamma/ExponentialMovingAverage/cond/Switch_1:0
?
cond_14/cond_textcond_14/pred_id:0cond_14/switch_t:0 *?
$Reconstruction_Output/dense/kernel:0
cond_14/pred_id:0
cond_14/read/Switch:1
cond_14/read:0
cond_14/switch_t:0=
$Reconstruction_Output/dense/kernel:0cond_14/read/Switch:1&
cond_14/pred_id:0cond_14/pred_id:0
?
cond_14/cond_text_1cond_14/pred_id:0cond_14/switch_f:0*?
?Reconstruction_Output/dense/kernel/Initializer/random_uniform:0
cond_14/Switch_1:0
cond_14/Switch_1:1
cond_14/pred_id:0
cond_14/switch_f:0U
?Reconstruction_Output/dense/kernel/Initializer/random_uniform:0cond_14/Switch_1:0&
cond_14/pred_id:0cond_14/pred_id:0
?
JReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/cond_textJReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/pred_id:0KReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/switch_t:0 *?
JReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/pred_id:0
NReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/read/Switch:1
GReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/read:0
KReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/switch_t:0
$Reconstruction_Output/dense/kernel:0?
JReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/pred_id:0JReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/pred_id:0v
$Reconstruction_Output/dense/kernel:0NReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/read/Switch:1
?
LReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/cond_text_1JReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/pred_id:0KReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/switch_f:0*?
KReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/Switch_1:0
KReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/Switch_1:1
JReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/pred_id:0
KReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/switch_f:0
?Reconstruction_Output/dense/kernel/Initializer/random_uniform:0?
JReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/pred_id:0JReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/pred_id:0?
?Reconstruction_Output/dense/kernel/Initializer/random_uniform:0KReconstruction_Output/dense/kernel/ExponentialMovingAverage/cond/Switch_1:0
?
cond_15/cond_textcond_15/pred_id:0cond_15/switch_t:0 *?
"Reconstruction_Output/dense/bias:0
cond_15/pred_id:0
cond_15/read/Switch:1
cond_15/read:0
cond_15/switch_t:0&
cond_15/pred_id:0cond_15/pred_id:0;
"Reconstruction_Output/dense/bias:0cond_15/read/Switch:1
?
cond_15/cond_text_1cond_15/pred_id:0cond_15/switch_f:0*?
4Reconstruction_Output/dense/bias/Initializer/zeros:0
cond_15/Switch_1:0
cond_15/Switch_1:1
cond_15/pred_id:0
cond_15/switch_f:0&
cond_15/pred_id:0cond_15/pred_id:0J
4Reconstruction_Output/dense/bias/Initializer/zeros:0cond_15/Switch_1:0
?
HReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/cond_textHReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/pred_id:0IReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/switch_t:0 *?
HReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/pred_id:0
LReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/read/Switch:1
EReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/read:0
IReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/switch_t:0
"Reconstruction_Output/dense/bias:0?
HReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/pred_id:0HReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/pred_id:0r
"Reconstruction_Output/dense/bias:0LReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/read/Switch:1
?
JReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/cond_text_1HReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/pred_id:0IReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/switch_f:0*?
IReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/Switch_1:0
IReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/Switch_1:1
HReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/pred_id:0
IReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/switch_f:0
4Reconstruction_Output/dense/bias/Initializer/zeros:0?
HReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/pred_id:0HReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/pred_id:0?
4Reconstruction_Output/dense/bias/Initializer/zeros:0IReconstruction_Output/dense/bias/ExponentialMovingAverage/cond/Switch_1:0
?
cond_16/cond_textcond_16/pred_id:0cond_16/switch_t:0 *?
cond_16/pred_id:0
cond_16/read/Switch:1
cond_16/read:0
cond_16/switch_t:0
dense/kernel:0&
cond_16/pred_id:0cond_16/pred_id:0'
dense/kernel:0cond_16/read/Switch:1
?
cond_16/cond_text_1cond_16/pred_id:0cond_16/switch_f:0*?
cond_16/Switch_1:0
cond_16/Switch_1:1
cond_16/pred_id:0
cond_16/switch_f:0
)dense/kernel/Initializer/random_uniform:0?
)dense/kernel/Initializer/random_uniform:0cond_16/Switch_1:0&
cond_16/pred_id:0cond_16/pred_id:0
?
4dense/kernel/ExponentialMovingAverage/cond/cond_text4dense/kernel/ExponentialMovingAverage/cond/pred_id:05dense/kernel/ExponentialMovingAverage/cond/switch_t:0 *?
4dense/kernel/ExponentialMovingAverage/cond/pred_id:0
8dense/kernel/ExponentialMovingAverage/cond/read/Switch:1
1dense/kernel/ExponentialMovingAverage/cond/read:0
5dense/kernel/ExponentialMovingAverage/cond/switch_t:0
dense/kernel:0l
4dense/kernel/ExponentialMovingAverage/cond/pred_id:04dense/kernel/ExponentialMovingAverage/cond/pred_id:0J
dense/kernel:08dense/kernel/ExponentialMovingAverage/cond/read/Switch:1
?
6dense/kernel/ExponentialMovingAverage/cond/cond_text_14dense/kernel/ExponentialMovingAverage/cond/pred_id:05dense/kernel/ExponentialMovingAverage/cond/switch_f:0*?
5dense/kernel/ExponentialMovingAverage/cond/Switch_1:0
5dense/kernel/ExponentialMovingAverage/cond/Switch_1:1
4dense/kernel/ExponentialMovingAverage/cond/pred_id:0
5dense/kernel/ExponentialMovingAverage/cond/switch_f:0
)dense/kernel/Initializer/random_uniform:0b
)dense/kernel/Initializer/random_uniform:05dense/kernel/ExponentialMovingAverage/cond/Switch_1:0l
4dense/kernel/ExponentialMovingAverage/cond/pred_id:04dense/kernel/ExponentialMovingAverage/cond/pred_id:0
?
cond_17/cond_textcond_17/pred_id:0cond_17/switch_t:0 *?
cond_17/pred_id:0
cond_17/read/Switch:1
cond_17/read:0
cond_17/switch_t:0
dense/bias:0%
dense/bias:0cond_17/read/Switch:1&
cond_17/pred_id:0cond_17/pred_id:0
?
cond_17/cond_text_1cond_17/pred_id:0cond_17/switch_f:0*?
cond_17/Switch_1:0
cond_17/Switch_1:1
cond_17/pred_id:0
cond_17/switch_f:0
dense/bias/Initializer/zeros:04
dense/bias/Initializer/zeros:0cond_17/Switch_1:0&
cond_17/pred_id:0cond_17/pred_id:0
?
2dense/bias/ExponentialMovingAverage/cond/cond_text2dense/bias/ExponentialMovingAverage/cond/pred_id:03dense/bias/ExponentialMovingAverage/cond/switch_t:0 *?
2dense/bias/ExponentialMovingAverage/cond/pred_id:0
6dense/bias/ExponentialMovingAverage/cond/read/Switch:1
/dense/bias/ExponentialMovingAverage/cond/read:0
3dense/bias/ExponentialMovingAverage/cond/switch_t:0
dense/bias:0h
2dense/bias/ExponentialMovingAverage/cond/pred_id:02dense/bias/ExponentialMovingAverage/cond/pred_id:0F
dense/bias:06dense/bias/ExponentialMovingAverage/cond/read/Switch:1
?
4dense/bias/ExponentialMovingAverage/cond/cond_text_12dense/bias/ExponentialMovingAverage/cond/pred_id:03dense/bias/ExponentialMovingAverage/cond/switch_f:0*?
3dense/bias/ExponentialMovingAverage/cond/Switch_1:0
3dense/bias/ExponentialMovingAverage/cond/Switch_1:1
2dense/bias/ExponentialMovingAverage/cond/pred_id:0
3dense/bias/ExponentialMovingAverage/cond/switch_f:0
dense/bias/Initializer/zeros:0U
dense/bias/Initializer/zeros:03dense/bias/ExponentialMovingAverage/cond/Switch_1:0h
2dense/bias/ExponentialMovingAverage/cond/pred_id:02dense/bias/ExponentialMovingAverage/cond/pred_id:0
?
cond_18/cond_textcond_18/pred_id:0cond_18/switch_t:0 *?
 FCU_muiltDense_x0/dense/kernel:0
cond_18/pred_id:0
cond_18/read/Switch:1
cond_18/read:0
cond_18/switch_t:0&
cond_18/pred_id:0cond_18/pred_id:09
 FCU_muiltDense_x0/dense/kernel:0cond_18/read/Switch:1
?
cond_18/cond_text_1cond_18/pred_id:0cond_18/switch_f:0*?
;FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform:0
cond_18/Switch_1:0
cond_18/Switch_1:1
cond_18/pred_id:0
cond_18/switch_f:0&
cond_18/pred_id:0cond_18/pred_id:0Q
;FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform:0cond_18/Switch_1:0
?
FFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/cond_textFFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/pred_id:0GFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/switch_t:0 *?
FFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/pred_id:0
JFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/read/Switch:1
CFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/read:0
GFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/switch_t:0
 FCU_muiltDense_x0/dense/kernel:0n
 FCU_muiltDense_x0/dense/kernel:0JFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/read/Switch:1?
FFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/pred_id:0FFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/pred_id:0
?
HFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/cond_text_1FFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/pred_id:0GFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/switch_f:0*?
GFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/Switch_1:0
GFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/Switch_1:1
FFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/pred_id:0
GFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/switch_f:0
;FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform:0?
;FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform:0GFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/Switch_1:0?
FFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/pred_id:0FFCU_muiltDense_x0/dense/kernel/ExponentialMovingAverage/cond/pred_id:0
?
cond_19/cond_textcond_19/pred_id:0cond_19/switch_t:0 *?
FCU_muiltDense_x0/dense/bias:0
cond_19/pred_id:0
cond_19/read/Switch:1
cond_19/read:0
cond_19/switch_t:0&
cond_19/pred_id:0cond_19/pred_id:07
FCU_muiltDense_x0/dense/bias:0cond_19/read/Switch:1
?
cond_19/cond_text_1cond_19/pred_id:0cond_19/switch_f:0*?
0FCU_muiltDense_x0/dense/bias/Initializer/zeros:0
cond_19/Switch_1:0
cond_19/Switch_1:1
cond_19/pred_id:0
cond_19/switch_f:0F
0FCU_muiltDense_x0/dense/bias/Initializer/zeros:0cond_19/Switch_1:0&
cond_19/pred_id:0cond_19/pred_id:0
?
DFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/cond_textDFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/pred_id:0EFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/switch_t:0 *?
DFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/pred_id:0
HFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/read/Switch:1
AFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/read:0
EFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/switch_t:0
FCU_muiltDense_x0/dense/bias:0j
FCU_muiltDense_x0/dense/bias:0HFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/read/Switch:1?
DFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/pred_id:0DFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/pred_id:0
?
FFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/cond_text_1DFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/pred_id:0EFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/switch_f:0*?
EFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/Switch_1:0
EFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/Switch_1:1
DFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/pred_id:0
EFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/switch_f:0
0FCU_muiltDense_x0/dense/bias/Initializer/zeros:0y
0FCU_muiltDense_x0/dense/bias/Initializer/zeros:0EFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/Switch_1:0?
DFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/pred_id:0DFCU_muiltDense_x0/dense/bias/ExponentialMovingAverage/cond/pred_id:0
?
cond_20/cond_textcond_20/pred_id:0cond_20/switch_t:0 *?
FCU_muiltDense_x0/beta:0
cond_20/pred_id:0
cond_20/read/Switch:1
cond_20/read:0
cond_20/switch_t:01
FCU_muiltDense_x0/beta:0cond_20/read/Switch:1&
cond_20/pred_id:0cond_20/pred_id:0
?
cond_20/cond_text_1cond_20/pred_id:0cond_20/switch_f:0*?
*FCU_muiltDense_x0/beta/Initializer/zeros:0
cond_20/Switch_1:0
cond_20/Switch_1:1
cond_20/pred_id:0
cond_20/switch_f:0&
cond_20/pred_id:0cond_20/pred_id:0@
*FCU_muiltDense_x0/beta/Initializer/zeros:0cond_20/Switch_1:0
?
>FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/cond_text>FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/pred_id:0?FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/switch_t:0 *?
>FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/pred_id:0
BFCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/read/Switch:1
;FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/read:0
?FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/switch_t:0
FCU_muiltDense_x0/beta:0^
FCU_muiltDense_x0/beta:0BFCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/read/Switch:1?
>FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/pred_id:0>FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/pred_id:0
?
@FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/cond_text_1>FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/pred_id:0?FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/switch_f:0*?
?FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/Switch_1:0
?FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/Switch_1:1
>FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/pred_id:0
?FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/switch_f:0
*FCU_muiltDense_x0/beta/Initializer/zeros:0m
*FCU_muiltDense_x0/beta/Initializer/zeros:0?FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/Switch_1:0?
>FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/pred_id:0>FCU_muiltDense_x0/beta/ExponentialMovingAverage/cond/pred_id:0
?
cond_21/cond_textcond_21/pred_id:0cond_21/switch_t:0 *?
FCU_muiltDense_x0/gamma:0
cond_21/pred_id:0
cond_21/read/Switch:1
cond_21/read:0
cond_21/switch_t:02
FCU_muiltDense_x0/gamma:0cond_21/read/Switch:1&
cond_21/pred_id:0cond_21/pred_id:0
?
cond_21/cond_text_1cond_21/pred_id:0cond_21/switch_f:0*?
*FCU_muiltDense_x0/gamma/Initializer/ones:0
cond_21/Switch_1:0
cond_21/Switch_1:1
cond_21/pred_id:0
cond_21/switch_f:0&
cond_21/pred_id:0cond_21/pred_id:0@
*FCU_muiltDense_x0/gamma/Initializer/ones:0cond_21/Switch_1:0
?
?FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/cond_text?FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/pred_id:0@FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/switch_t:0 *?
?FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/pred_id:0
CFCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/read/Switch:1
<FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/read:0
@FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/switch_t:0
FCU_muiltDense_x0/gamma:0`
FCU_muiltDense_x0/gamma:0CFCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/read/Switch:1?
?FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/pred_id:0?FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/pred_id:0
?
AFCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/cond_text_1?FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/pred_id:0@FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/switch_f:0*?
@FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/Switch_1:0
@FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/Switch_1:1
?FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/pred_id:0
@FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/switch_f:0
*FCU_muiltDense_x0/gamma/Initializer/ones:0?
?FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/pred_id:0?FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/pred_id:0n
*FCU_muiltDense_x0/gamma/Initializer/ones:0@FCU_muiltDense_x0/gamma/ExponentialMovingAverage/cond/Switch_1:0
?
cond_22/cond_textcond_22/pred_id:0cond_22/switch_t:0 *?
Output_/dense/kernel:0
cond_22/pred_id:0
cond_22/read/Switch:1
cond_22/read:0
cond_22/switch_t:0&
cond_22/pred_id:0cond_22/pred_id:0/
Output_/dense/kernel:0cond_22/read/Switch:1
?
cond_22/cond_text_1cond_22/pred_id:0cond_22/switch_f:0*?
1Output_/dense/kernel/Initializer/random_uniform:0
cond_22/Switch_1:0
cond_22/Switch_1:1
cond_22/pred_id:0
cond_22/switch_f:0G
1Output_/dense/kernel/Initializer/random_uniform:0cond_22/Switch_1:0&
cond_22/pred_id:0cond_22/pred_id:0
?
<Output_/dense/kernel/ExponentialMovingAverage/cond/cond_text<Output_/dense/kernel/ExponentialMovingAverage/cond/pred_id:0=Output_/dense/kernel/ExponentialMovingAverage/cond/switch_t:0 *?
<Output_/dense/kernel/ExponentialMovingAverage/cond/pred_id:0
@Output_/dense/kernel/ExponentialMovingAverage/cond/read/Switch:1
9Output_/dense/kernel/ExponentialMovingAverage/cond/read:0
=Output_/dense/kernel/ExponentialMovingAverage/cond/switch_t:0
Output_/dense/kernel:0Z
Output_/dense/kernel:0@Output_/dense/kernel/ExponentialMovingAverage/cond/read/Switch:1|
<Output_/dense/kernel/ExponentialMovingAverage/cond/pred_id:0<Output_/dense/kernel/ExponentialMovingAverage/cond/pred_id:0
?
>Output_/dense/kernel/ExponentialMovingAverage/cond/cond_text_1<Output_/dense/kernel/ExponentialMovingAverage/cond/pred_id:0=Output_/dense/kernel/ExponentialMovingAverage/cond/switch_f:0*?
=Output_/dense/kernel/ExponentialMovingAverage/cond/Switch_1:0
=Output_/dense/kernel/ExponentialMovingAverage/cond/Switch_1:1
<Output_/dense/kernel/ExponentialMovingAverage/cond/pred_id:0
=Output_/dense/kernel/ExponentialMovingAverage/cond/switch_f:0
1Output_/dense/kernel/Initializer/random_uniform:0r
1Output_/dense/kernel/Initializer/random_uniform:0=Output_/dense/kernel/ExponentialMovingAverage/cond/Switch_1:0|
<Output_/dense/kernel/ExponentialMovingAverage/cond/pred_id:0<Output_/dense/kernel/ExponentialMovingAverage/cond/pred_id:0
?
cond_23/cond_textcond_23/pred_id:0cond_23/switch_t:0 *?
Output_/dense/bias:0
cond_23/pred_id:0
cond_23/read/Switch:1
cond_23/read:0
cond_23/switch_t:0&
cond_23/pred_id:0cond_23/pred_id:0-
Output_/dense/bias:0cond_23/read/Switch:1
?
cond_23/cond_text_1cond_23/pred_id:0cond_23/switch_f:0*?
&Output_/dense/bias/Initializer/zeros:0
cond_23/Switch_1:0
cond_23/Switch_1:1
cond_23/pred_id:0
cond_23/switch_f:0&
cond_23/pred_id:0cond_23/pred_id:0<
&Output_/dense/bias/Initializer/zeros:0cond_23/Switch_1:0
?
:Output_/dense/bias/ExponentialMovingAverage/cond/cond_text:Output_/dense/bias/ExponentialMovingAverage/cond/pred_id:0;Output_/dense/bias/ExponentialMovingAverage/cond/switch_t:0 *?
:Output_/dense/bias/ExponentialMovingAverage/cond/pred_id:0
>Output_/dense/bias/ExponentialMovingAverage/cond/read/Switch:1
7Output_/dense/bias/ExponentialMovingAverage/cond/read:0
;Output_/dense/bias/ExponentialMovingAverage/cond/switch_t:0
Output_/dense/bias:0x
:Output_/dense/bias/ExponentialMovingAverage/cond/pred_id:0:Output_/dense/bias/ExponentialMovingAverage/cond/pred_id:0V
Output_/dense/bias:0>Output_/dense/bias/ExponentialMovingAverage/cond/read/Switch:1
?
<Output_/dense/bias/ExponentialMovingAverage/cond/cond_text_1:Output_/dense/bias/ExponentialMovingAverage/cond/pred_id:0;Output_/dense/bias/ExponentialMovingAverage/cond/switch_f:0*?
;Output_/dense/bias/ExponentialMovingAverage/cond/Switch_1:0
;Output_/dense/bias/ExponentialMovingAverage/cond/Switch_1:1
:Output_/dense/bias/ExponentialMovingAverage/cond/pred_id:0
;Output_/dense/bias/ExponentialMovingAverage/cond/switch_f:0
&Output_/dense/bias/Initializer/zeros:0e
&Output_/dense/bias/Initializer/zeros:0;Output_/dense/bias/ExponentialMovingAverage/cond/Switch_1:0x
:Output_/dense/bias/ExponentialMovingAverage/cond/pred_id:0:Output_/dense/bias/ExponentialMovingAverage/cond/pred_id:0
?
BackupVariables/cond/cond_textBackupVariables/cond/pred_id:0BackupVariables/cond/switch_t:0 *?
BackupVariables/cond/pred_id:0
"BackupVariables/cond/read/Switch:1
BackupVariables/cond/read:0
BackupVariables/cond/switch_t:0
CCN_1Conv_x0/convA10/kernel:0C
CCN_1Conv_x0/convA10/kernel:0"BackupVariables/cond/read/Switch:1@
BackupVariables/cond/pred_id:0BackupVariables/cond/pred_id:0
?
 BackupVariables/cond/cond_text_1BackupVariables/cond/pred_id:0BackupVariables/cond/switch_f:0*?
BackupVariables/cond/Switch_1:0
BackupVariables/cond/Switch_1:1
BackupVariables/cond/pred_id:0
BackupVariables/cond/switch_f:0
8CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform:0[
8CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform:0BackupVariables/cond/Switch_1:0@
BackupVariables/cond/pred_id:0BackupVariables/cond/pred_id:0
?
:BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/cond_text:BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/switch_t:0 *?
:BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/pred_id:0
>BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/read/Switch:1
7BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/read:0
;BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/switch_t:0
CCN_1Conv_x0/convA10/kernel:0x
:BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/pred_id:0_
CCN_1Conv_x0/convA10/kernel:0>BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/read/Switch:1
?
<BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/cond_text_1:BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/switch_f:0*?
;BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/Switch_1:0
;BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/Switch_1:1
:BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/pred_id:0
;BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/switch_f:0
8CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform:0x
:BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/pred_id:0w
8CCN_1Conv_x0/convA10/kernel/Initializer/random_uniform:0;BackupVariables/CCN_1Conv_x0/convA10/kernel/cond/Switch_1:0
?
 BackupVariables/cond_1/cond_text BackupVariables/cond_1/pred_id:0!BackupVariables/cond_1/switch_t:0 *?
 BackupVariables/cond_1/pred_id:0
$BackupVariables/cond_1/read/Switch:1
BackupVariables/cond_1/read:0
!BackupVariables/cond_1/switch_t:0
CCN_1Conv_x0/convA10/bias:0D
 BackupVariables/cond_1/pred_id:0 BackupVariables/cond_1/pred_id:0C
CCN_1Conv_x0/convA10/bias:0$BackupVariables/cond_1/read/Switch:1
?
"BackupVariables/cond_1/cond_text_1 BackupVariables/cond_1/pred_id:0!BackupVariables/cond_1/switch_f:0*?
!BackupVariables/cond_1/Switch_1:0
!BackupVariables/cond_1/Switch_1:1
 BackupVariables/cond_1/pred_id:0
!BackupVariables/cond_1/switch_f:0
-CCN_1Conv_x0/convA10/bias/Initializer/zeros:0R
-CCN_1Conv_x0/convA10/bias/Initializer/zeros:0!BackupVariables/cond_1/Switch_1:0D
 BackupVariables/cond_1/pred_id:0 BackupVariables/cond_1/pred_id:0
?
8BackupVariables/CCN_1Conv_x0/convA10/bias/cond/cond_text8BackupVariables/CCN_1Conv_x0/convA10/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convA10/bias/cond/switch_t:0 *?
8BackupVariables/CCN_1Conv_x0/convA10/bias/cond/pred_id:0
<BackupVariables/CCN_1Conv_x0/convA10/bias/cond/read/Switch:1
5BackupVariables/CCN_1Conv_x0/convA10/bias/cond/read:0
9BackupVariables/CCN_1Conv_x0/convA10/bias/cond/switch_t:0
CCN_1Conv_x0/convA10/bias:0t
8BackupVariables/CCN_1Conv_x0/convA10/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convA10/bias/cond/pred_id:0[
CCN_1Conv_x0/convA10/bias:0<BackupVariables/CCN_1Conv_x0/convA10/bias/cond/read/Switch:1
?
:BackupVariables/CCN_1Conv_x0/convA10/bias/cond/cond_text_18BackupVariables/CCN_1Conv_x0/convA10/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convA10/bias/cond/switch_f:0*?
9BackupVariables/CCN_1Conv_x0/convA10/bias/cond/Switch_1:0
9BackupVariables/CCN_1Conv_x0/convA10/bias/cond/Switch_1:1
8BackupVariables/CCN_1Conv_x0/convA10/bias/cond/pred_id:0
9BackupVariables/CCN_1Conv_x0/convA10/bias/cond/switch_f:0
-CCN_1Conv_x0/convA10/bias/Initializer/zeros:0t
8BackupVariables/CCN_1Conv_x0/convA10/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convA10/bias/cond/pred_id:0j
-CCN_1Conv_x0/convA10/bias/Initializer/zeros:09BackupVariables/CCN_1Conv_x0/convA10/bias/cond/Switch_1:0
?
 BackupVariables/cond_2/cond_text BackupVariables/cond_2/pred_id:0!BackupVariables/cond_2/switch_t:0 *?
 BackupVariables/cond_2/pred_id:0
$BackupVariables/cond_2/read/Switch:1
BackupVariables/cond_2/read:0
!BackupVariables/cond_2/switch_t:0
CCN_1Conv_x0/convB10/kernel:0E
CCN_1Conv_x0/convB10/kernel:0$BackupVariables/cond_2/read/Switch:1D
 BackupVariables/cond_2/pred_id:0 BackupVariables/cond_2/pred_id:0
?
"BackupVariables/cond_2/cond_text_1 BackupVariables/cond_2/pred_id:0!BackupVariables/cond_2/switch_f:0*?
!BackupVariables/cond_2/Switch_1:0
!BackupVariables/cond_2/Switch_1:1
 BackupVariables/cond_2/pred_id:0
!BackupVariables/cond_2/switch_f:0
8CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform:0]
8CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform:0!BackupVariables/cond_2/Switch_1:0D
 BackupVariables/cond_2/pred_id:0 BackupVariables/cond_2/pred_id:0
?
:BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/cond_text:BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/switch_t:0 *?
:BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/pred_id:0
>BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/read/Switch:1
7BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/read:0
;BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/switch_t:0
CCN_1Conv_x0/convB10/kernel:0x
:BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/pred_id:0_
CCN_1Conv_x0/convB10/kernel:0>BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/read/Switch:1
?
<BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/cond_text_1:BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/switch_f:0*?
;BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/Switch_1:0
;BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/Switch_1:1
:BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/pred_id:0
;BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/switch_f:0
8CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform:0w
8CCN_1Conv_x0/convB10/kernel/Initializer/random_uniform:0;BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/Switch_1:0x
:BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convB10/kernel/cond/pred_id:0
?
 BackupVariables/cond_3/cond_text BackupVariables/cond_3/pred_id:0!BackupVariables/cond_3/switch_t:0 *?
 BackupVariables/cond_3/pred_id:0
$BackupVariables/cond_3/read/Switch:1
BackupVariables/cond_3/read:0
!BackupVariables/cond_3/switch_t:0
CCN_1Conv_x0/convB10/bias:0C
CCN_1Conv_x0/convB10/bias:0$BackupVariables/cond_3/read/Switch:1D
 BackupVariables/cond_3/pred_id:0 BackupVariables/cond_3/pred_id:0
?
"BackupVariables/cond_3/cond_text_1 BackupVariables/cond_3/pred_id:0!BackupVariables/cond_3/switch_f:0*?
!BackupVariables/cond_3/Switch_1:0
!BackupVariables/cond_3/Switch_1:1
 BackupVariables/cond_3/pred_id:0
!BackupVariables/cond_3/switch_f:0
-CCN_1Conv_x0/convB10/bias/Initializer/zeros:0D
 BackupVariables/cond_3/pred_id:0 BackupVariables/cond_3/pred_id:0R
-CCN_1Conv_x0/convB10/bias/Initializer/zeros:0!BackupVariables/cond_3/Switch_1:0
?
8BackupVariables/CCN_1Conv_x0/convB10/bias/cond/cond_text8BackupVariables/CCN_1Conv_x0/convB10/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convB10/bias/cond/switch_t:0 *?
8BackupVariables/CCN_1Conv_x0/convB10/bias/cond/pred_id:0
<BackupVariables/CCN_1Conv_x0/convB10/bias/cond/read/Switch:1
5BackupVariables/CCN_1Conv_x0/convB10/bias/cond/read:0
9BackupVariables/CCN_1Conv_x0/convB10/bias/cond/switch_t:0
CCN_1Conv_x0/convB10/bias:0[
CCN_1Conv_x0/convB10/bias:0<BackupVariables/CCN_1Conv_x0/convB10/bias/cond/read/Switch:1t
8BackupVariables/CCN_1Conv_x0/convB10/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convB10/bias/cond/pred_id:0
?
:BackupVariables/CCN_1Conv_x0/convB10/bias/cond/cond_text_18BackupVariables/CCN_1Conv_x0/convB10/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convB10/bias/cond/switch_f:0*?
9BackupVariables/CCN_1Conv_x0/convB10/bias/cond/Switch_1:0
9BackupVariables/CCN_1Conv_x0/convB10/bias/cond/Switch_1:1
8BackupVariables/CCN_1Conv_x0/convB10/bias/cond/pred_id:0
9BackupVariables/CCN_1Conv_x0/convB10/bias/cond/switch_f:0
-CCN_1Conv_x0/convB10/bias/Initializer/zeros:0j
-CCN_1Conv_x0/convB10/bias/Initializer/zeros:09BackupVariables/CCN_1Conv_x0/convB10/bias/cond/Switch_1:0t
8BackupVariables/CCN_1Conv_x0/convB10/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convB10/bias/cond/pred_id:0
?
 BackupVariables/cond_4/cond_text BackupVariables/cond_4/pred_id:0!BackupVariables/cond_4/switch_t:0 *?
 BackupVariables/cond_4/pred_id:0
$BackupVariables/cond_4/read/Switch:1
BackupVariables/cond_4/read:0
!BackupVariables/cond_4/switch_t:0
CCN_1Conv_x0/convB20/kernel:0E
CCN_1Conv_x0/convB20/kernel:0$BackupVariables/cond_4/read/Switch:1D
 BackupVariables/cond_4/pred_id:0 BackupVariables/cond_4/pred_id:0
?
"BackupVariables/cond_4/cond_text_1 BackupVariables/cond_4/pred_id:0!BackupVariables/cond_4/switch_f:0*?
!BackupVariables/cond_4/Switch_1:0
!BackupVariables/cond_4/Switch_1:1
 BackupVariables/cond_4/pred_id:0
!BackupVariables/cond_4/switch_f:0
8CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform:0D
 BackupVariables/cond_4/pred_id:0 BackupVariables/cond_4/pred_id:0]
8CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform:0!BackupVariables/cond_4/Switch_1:0
?
:BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/cond_text:BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/switch_t:0 *?
:BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/pred_id:0
>BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/read/Switch:1
7BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/read:0
;BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/switch_t:0
CCN_1Conv_x0/convB20/kernel:0x
:BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/pred_id:0_
CCN_1Conv_x0/convB20/kernel:0>BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/read/Switch:1
?
<BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/cond_text_1:BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/switch_f:0*?
;BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/Switch_1:0
;BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/Switch_1:1
:BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/pred_id:0
;BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/switch_f:0
8CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform:0w
8CCN_1Conv_x0/convB20/kernel/Initializer/random_uniform:0;BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/Switch_1:0x
:BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convB20/kernel/cond/pred_id:0
?
 BackupVariables/cond_5/cond_text BackupVariables/cond_5/pred_id:0!BackupVariables/cond_5/switch_t:0 *?
 BackupVariables/cond_5/pred_id:0
$BackupVariables/cond_5/read/Switch:1
BackupVariables/cond_5/read:0
!BackupVariables/cond_5/switch_t:0
CCN_1Conv_x0/convB20/bias:0C
CCN_1Conv_x0/convB20/bias:0$BackupVariables/cond_5/read/Switch:1D
 BackupVariables/cond_5/pred_id:0 BackupVariables/cond_5/pred_id:0
?
"BackupVariables/cond_5/cond_text_1 BackupVariables/cond_5/pred_id:0!BackupVariables/cond_5/switch_f:0*?
!BackupVariables/cond_5/Switch_1:0
!BackupVariables/cond_5/Switch_1:1
 BackupVariables/cond_5/pred_id:0
!BackupVariables/cond_5/switch_f:0
-CCN_1Conv_x0/convB20/bias/Initializer/zeros:0D
 BackupVariables/cond_5/pred_id:0 BackupVariables/cond_5/pred_id:0R
-CCN_1Conv_x0/convB20/bias/Initializer/zeros:0!BackupVariables/cond_5/Switch_1:0
?
8BackupVariables/CCN_1Conv_x0/convB20/bias/cond/cond_text8BackupVariables/CCN_1Conv_x0/convB20/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convB20/bias/cond/switch_t:0 *?
8BackupVariables/CCN_1Conv_x0/convB20/bias/cond/pred_id:0
<BackupVariables/CCN_1Conv_x0/convB20/bias/cond/read/Switch:1
5BackupVariables/CCN_1Conv_x0/convB20/bias/cond/read:0
9BackupVariables/CCN_1Conv_x0/convB20/bias/cond/switch_t:0
CCN_1Conv_x0/convB20/bias:0[
CCN_1Conv_x0/convB20/bias:0<BackupVariables/CCN_1Conv_x0/convB20/bias/cond/read/Switch:1t
8BackupVariables/CCN_1Conv_x0/convB20/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convB20/bias/cond/pred_id:0
?
:BackupVariables/CCN_1Conv_x0/convB20/bias/cond/cond_text_18BackupVariables/CCN_1Conv_x0/convB20/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convB20/bias/cond/switch_f:0*?
9BackupVariables/CCN_1Conv_x0/convB20/bias/cond/Switch_1:0
9BackupVariables/CCN_1Conv_x0/convB20/bias/cond/Switch_1:1
8BackupVariables/CCN_1Conv_x0/convB20/bias/cond/pred_id:0
9BackupVariables/CCN_1Conv_x0/convB20/bias/cond/switch_f:0
-CCN_1Conv_x0/convB20/bias/Initializer/zeros:0t
8BackupVariables/CCN_1Conv_x0/convB20/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convB20/bias/cond/pred_id:0j
-CCN_1Conv_x0/convB20/bias/Initializer/zeros:09BackupVariables/CCN_1Conv_x0/convB20/bias/cond/Switch_1:0
?
 BackupVariables/cond_6/cond_text BackupVariables/cond_6/pred_id:0!BackupVariables/cond_6/switch_t:0 *?
 BackupVariables/cond_6/pred_id:0
$BackupVariables/cond_6/read/Switch:1
BackupVariables/cond_6/read:0
!BackupVariables/cond_6/switch_t:0
CCN_1Conv_x0/convA11/kernel:0D
 BackupVariables/cond_6/pred_id:0 BackupVariables/cond_6/pred_id:0E
CCN_1Conv_x0/convA11/kernel:0$BackupVariables/cond_6/read/Switch:1
?
"BackupVariables/cond_6/cond_text_1 BackupVariables/cond_6/pred_id:0!BackupVariables/cond_6/switch_f:0*?
!BackupVariables/cond_6/Switch_1:0
!BackupVariables/cond_6/Switch_1:1
 BackupVariables/cond_6/pred_id:0
!BackupVariables/cond_6/switch_f:0
8CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform:0]
8CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform:0!BackupVariables/cond_6/Switch_1:0D
 BackupVariables/cond_6/pred_id:0 BackupVariables/cond_6/pred_id:0
?
:BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/cond_text:BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/switch_t:0 *?
:BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/pred_id:0
>BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/read/Switch:1
7BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/read:0
;BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/switch_t:0
CCN_1Conv_x0/convA11/kernel:0_
CCN_1Conv_x0/convA11/kernel:0>BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/read/Switch:1x
:BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/pred_id:0
?
<BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/cond_text_1:BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/switch_f:0*?
;BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/Switch_1:0
;BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/Switch_1:1
:BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/pred_id:0
;BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/switch_f:0
8CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform:0x
:BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/pred_id:0w
8CCN_1Conv_x0/convA11/kernel/Initializer/random_uniform:0;BackupVariables/CCN_1Conv_x0/convA11/kernel/cond/Switch_1:0
?
 BackupVariables/cond_7/cond_text BackupVariables/cond_7/pred_id:0!BackupVariables/cond_7/switch_t:0 *?
 BackupVariables/cond_7/pred_id:0
$BackupVariables/cond_7/read/Switch:1
BackupVariables/cond_7/read:0
!BackupVariables/cond_7/switch_t:0
CCN_1Conv_x0/convA11/bias:0C
CCN_1Conv_x0/convA11/bias:0$BackupVariables/cond_7/read/Switch:1D
 BackupVariables/cond_7/pred_id:0 BackupVariables/cond_7/pred_id:0
?
"BackupVariables/cond_7/cond_text_1 BackupVariables/cond_7/pred_id:0!BackupVariables/cond_7/switch_f:0*?
!BackupVariables/cond_7/Switch_1:0
!BackupVariables/cond_7/Switch_1:1
 BackupVariables/cond_7/pred_id:0
!BackupVariables/cond_7/switch_f:0
-CCN_1Conv_x0/convA11/bias/Initializer/zeros:0D
 BackupVariables/cond_7/pred_id:0 BackupVariables/cond_7/pred_id:0R
-CCN_1Conv_x0/convA11/bias/Initializer/zeros:0!BackupVariables/cond_7/Switch_1:0
?
8BackupVariables/CCN_1Conv_x0/convA11/bias/cond/cond_text8BackupVariables/CCN_1Conv_x0/convA11/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convA11/bias/cond/switch_t:0 *?
8BackupVariables/CCN_1Conv_x0/convA11/bias/cond/pred_id:0
<BackupVariables/CCN_1Conv_x0/convA11/bias/cond/read/Switch:1
5BackupVariables/CCN_1Conv_x0/convA11/bias/cond/read:0
9BackupVariables/CCN_1Conv_x0/convA11/bias/cond/switch_t:0
CCN_1Conv_x0/convA11/bias:0t
8BackupVariables/CCN_1Conv_x0/convA11/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convA11/bias/cond/pred_id:0[
CCN_1Conv_x0/convA11/bias:0<BackupVariables/CCN_1Conv_x0/convA11/bias/cond/read/Switch:1
?
:BackupVariables/CCN_1Conv_x0/convA11/bias/cond/cond_text_18BackupVariables/CCN_1Conv_x0/convA11/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convA11/bias/cond/switch_f:0*?
9BackupVariables/CCN_1Conv_x0/convA11/bias/cond/Switch_1:0
9BackupVariables/CCN_1Conv_x0/convA11/bias/cond/Switch_1:1
8BackupVariables/CCN_1Conv_x0/convA11/bias/cond/pred_id:0
9BackupVariables/CCN_1Conv_x0/convA11/bias/cond/switch_f:0
-CCN_1Conv_x0/convA11/bias/Initializer/zeros:0j
-CCN_1Conv_x0/convA11/bias/Initializer/zeros:09BackupVariables/CCN_1Conv_x0/convA11/bias/cond/Switch_1:0t
8BackupVariables/CCN_1Conv_x0/convA11/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convA11/bias/cond/pred_id:0
?
 BackupVariables/cond_8/cond_text BackupVariables/cond_8/pred_id:0!BackupVariables/cond_8/switch_t:0 *?
 BackupVariables/cond_8/pred_id:0
$BackupVariables/cond_8/read/Switch:1
BackupVariables/cond_8/read:0
!BackupVariables/cond_8/switch_t:0
CCN_1Conv_x0/convB11/kernel:0E
CCN_1Conv_x0/convB11/kernel:0$BackupVariables/cond_8/read/Switch:1D
 BackupVariables/cond_8/pred_id:0 BackupVariables/cond_8/pred_id:0
?
"BackupVariables/cond_8/cond_text_1 BackupVariables/cond_8/pred_id:0!BackupVariables/cond_8/switch_f:0*?
!BackupVariables/cond_8/Switch_1:0
!BackupVariables/cond_8/Switch_1:1
 BackupVariables/cond_8/pred_id:0
!BackupVariables/cond_8/switch_f:0
8CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform:0]
8CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform:0!BackupVariables/cond_8/Switch_1:0D
 BackupVariables/cond_8/pred_id:0 BackupVariables/cond_8/pred_id:0
?
:BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/cond_text:BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/switch_t:0 *?
:BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/pred_id:0
>BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/read/Switch:1
7BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/read:0
;BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/switch_t:0
CCN_1Conv_x0/convB11/kernel:0x
:BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/pred_id:0_
CCN_1Conv_x0/convB11/kernel:0>BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/read/Switch:1
?
<BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/cond_text_1:BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/switch_f:0*?
;BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/Switch_1:0
;BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/Switch_1:1
:BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/pred_id:0
;BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/switch_f:0
8CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform:0w
8CCN_1Conv_x0/convB11/kernel/Initializer/random_uniform:0;BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/Switch_1:0x
:BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convB11/kernel/cond/pred_id:0
?
 BackupVariables/cond_9/cond_text BackupVariables/cond_9/pred_id:0!BackupVariables/cond_9/switch_t:0 *?
 BackupVariables/cond_9/pred_id:0
$BackupVariables/cond_9/read/Switch:1
BackupVariables/cond_9/read:0
!BackupVariables/cond_9/switch_t:0
CCN_1Conv_x0/convB11/bias:0D
 BackupVariables/cond_9/pred_id:0 BackupVariables/cond_9/pred_id:0C
CCN_1Conv_x0/convB11/bias:0$BackupVariables/cond_9/read/Switch:1
?
"BackupVariables/cond_9/cond_text_1 BackupVariables/cond_9/pred_id:0!BackupVariables/cond_9/switch_f:0*?
!BackupVariables/cond_9/Switch_1:0
!BackupVariables/cond_9/Switch_1:1
 BackupVariables/cond_9/pred_id:0
!BackupVariables/cond_9/switch_f:0
-CCN_1Conv_x0/convB11/bias/Initializer/zeros:0D
 BackupVariables/cond_9/pred_id:0 BackupVariables/cond_9/pred_id:0R
-CCN_1Conv_x0/convB11/bias/Initializer/zeros:0!BackupVariables/cond_9/Switch_1:0
?
8BackupVariables/CCN_1Conv_x0/convB11/bias/cond/cond_text8BackupVariables/CCN_1Conv_x0/convB11/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convB11/bias/cond/switch_t:0 *?
8BackupVariables/CCN_1Conv_x0/convB11/bias/cond/pred_id:0
<BackupVariables/CCN_1Conv_x0/convB11/bias/cond/read/Switch:1
5BackupVariables/CCN_1Conv_x0/convB11/bias/cond/read:0
9BackupVariables/CCN_1Conv_x0/convB11/bias/cond/switch_t:0
CCN_1Conv_x0/convB11/bias:0t
8BackupVariables/CCN_1Conv_x0/convB11/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convB11/bias/cond/pred_id:0[
CCN_1Conv_x0/convB11/bias:0<BackupVariables/CCN_1Conv_x0/convB11/bias/cond/read/Switch:1
?
:BackupVariables/CCN_1Conv_x0/convB11/bias/cond/cond_text_18BackupVariables/CCN_1Conv_x0/convB11/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convB11/bias/cond/switch_f:0*?
9BackupVariables/CCN_1Conv_x0/convB11/bias/cond/Switch_1:0
9BackupVariables/CCN_1Conv_x0/convB11/bias/cond/Switch_1:1
8BackupVariables/CCN_1Conv_x0/convB11/bias/cond/pred_id:0
9BackupVariables/CCN_1Conv_x0/convB11/bias/cond/switch_f:0
-CCN_1Conv_x0/convB11/bias/Initializer/zeros:0t
8BackupVariables/CCN_1Conv_x0/convB11/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convB11/bias/cond/pred_id:0j
-CCN_1Conv_x0/convB11/bias/Initializer/zeros:09BackupVariables/CCN_1Conv_x0/convB11/bias/cond/Switch_1:0
?
!BackupVariables/cond_10/cond_text!BackupVariables/cond_10/pred_id:0"BackupVariables/cond_10/switch_t:0 *?
!BackupVariables/cond_10/pred_id:0
%BackupVariables/cond_10/read/Switch:1
BackupVariables/cond_10/read:0
"BackupVariables/cond_10/switch_t:0
CCN_1Conv_x0/convB21/kernel:0F
CCN_1Conv_x0/convB21/kernel:0%BackupVariables/cond_10/read/Switch:1F
!BackupVariables/cond_10/pred_id:0!BackupVariables/cond_10/pred_id:0
?
#BackupVariables/cond_10/cond_text_1!BackupVariables/cond_10/pred_id:0"BackupVariables/cond_10/switch_f:0*?
"BackupVariables/cond_10/Switch_1:0
"BackupVariables/cond_10/Switch_1:1
!BackupVariables/cond_10/pred_id:0
"BackupVariables/cond_10/switch_f:0
8CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform:0F
!BackupVariables/cond_10/pred_id:0!BackupVariables/cond_10/pred_id:0^
8CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform:0"BackupVariables/cond_10/Switch_1:0
?
:BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/cond_text:BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/switch_t:0 *?
:BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/pred_id:0
>BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/read/Switch:1
7BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/read:0
;BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/switch_t:0
CCN_1Conv_x0/convB21/kernel:0x
:BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/pred_id:0_
CCN_1Conv_x0/convB21/kernel:0>BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/read/Switch:1
?
<BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/cond_text_1:BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/pred_id:0;BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/switch_f:0*?
;BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/Switch_1:0
;BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/Switch_1:1
:BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/pred_id:0
;BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/switch_f:0
8CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform:0w
8CCN_1Conv_x0/convB21/kernel/Initializer/random_uniform:0;BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/Switch_1:0x
:BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/pred_id:0:BackupVariables/CCN_1Conv_x0/convB21/kernel/cond/pred_id:0
?
!BackupVariables/cond_11/cond_text!BackupVariables/cond_11/pred_id:0"BackupVariables/cond_11/switch_t:0 *?
!BackupVariables/cond_11/pred_id:0
%BackupVariables/cond_11/read/Switch:1
BackupVariables/cond_11/read:0
"BackupVariables/cond_11/switch_t:0
CCN_1Conv_x0/convB21/bias:0F
!BackupVariables/cond_11/pred_id:0!BackupVariables/cond_11/pred_id:0D
CCN_1Conv_x0/convB21/bias:0%BackupVariables/cond_11/read/Switch:1
?
#BackupVariables/cond_11/cond_text_1!BackupVariables/cond_11/pred_id:0"BackupVariables/cond_11/switch_f:0*?
"BackupVariables/cond_11/Switch_1:0
"BackupVariables/cond_11/Switch_1:1
!BackupVariables/cond_11/pred_id:0
"BackupVariables/cond_11/switch_f:0
-CCN_1Conv_x0/convB21/bias/Initializer/zeros:0S
-CCN_1Conv_x0/convB21/bias/Initializer/zeros:0"BackupVariables/cond_11/Switch_1:0F
!BackupVariables/cond_11/pred_id:0!BackupVariables/cond_11/pred_id:0
?
8BackupVariables/CCN_1Conv_x0/convB21/bias/cond/cond_text8BackupVariables/CCN_1Conv_x0/convB21/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convB21/bias/cond/switch_t:0 *?
8BackupVariables/CCN_1Conv_x0/convB21/bias/cond/pred_id:0
<BackupVariables/CCN_1Conv_x0/convB21/bias/cond/read/Switch:1
5BackupVariables/CCN_1Conv_x0/convB21/bias/cond/read:0
9BackupVariables/CCN_1Conv_x0/convB21/bias/cond/switch_t:0
CCN_1Conv_x0/convB21/bias:0[
CCN_1Conv_x0/convB21/bias:0<BackupVariables/CCN_1Conv_x0/convB21/bias/cond/read/Switch:1t
8BackupVariables/CCN_1Conv_x0/convB21/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convB21/bias/cond/pred_id:0
?
:BackupVariables/CCN_1Conv_x0/convB21/bias/cond/cond_text_18BackupVariables/CCN_1Conv_x0/convB21/bias/cond/pred_id:09BackupVariables/CCN_1Conv_x0/convB21/bias/cond/switch_f:0*?
9BackupVariables/CCN_1Conv_x0/convB21/bias/cond/Switch_1:0
9BackupVariables/CCN_1Conv_x0/convB21/bias/cond/Switch_1:1
8BackupVariables/CCN_1Conv_x0/convB21/bias/cond/pred_id:0
9BackupVariables/CCN_1Conv_x0/convB21/bias/cond/switch_f:0
-CCN_1Conv_x0/convB21/bias/Initializer/zeros:0t
8BackupVariables/CCN_1Conv_x0/convB21/bias/cond/pred_id:08BackupVariables/CCN_1Conv_x0/convB21/bias/cond/pred_id:0j
-CCN_1Conv_x0/convB21/bias/Initializer/zeros:09BackupVariables/CCN_1Conv_x0/convB21/bias/cond/Switch_1:0
?
!BackupVariables/cond_12/cond_text!BackupVariables/cond_12/pred_id:0"BackupVariables/cond_12/switch_t:0 *?
!BackupVariables/cond_12/pred_id:0
%BackupVariables/cond_12/read/Switch:1
BackupVariables/cond_12/read:0
"BackupVariables/cond_12/switch_t:0
Conv_out__/beta:0F
!BackupVariables/cond_12/pred_id:0!BackupVariables/cond_12/pred_id:0:
Conv_out__/beta:0%BackupVariables/cond_12/read/Switch:1
?
#BackupVariables/cond_12/cond_text_1!BackupVariables/cond_12/pred_id:0"BackupVariables/cond_12/switch_f:0*?
"BackupVariables/cond_12/Switch_1:0
"BackupVariables/cond_12/Switch_1:1
!BackupVariables/cond_12/pred_id:0
"BackupVariables/cond_12/switch_f:0
#Conv_out__/beta/Initializer/zeros:0I
#Conv_out__/beta/Initializer/zeros:0"BackupVariables/cond_12/Switch_1:0F
!BackupVariables/cond_12/pred_id:0!BackupVariables/cond_12/pred_id:0
?
.BackupVariables/Conv_out__/beta/cond/cond_text.BackupVariables/Conv_out__/beta/cond/pred_id:0/BackupVariables/Conv_out__/beta/cond/switch_t:0 *?
.BackupVariables/Conv_out__/beta/cond/pred_id:0
2BackupVariables/Conv_out__/beta/cond/read/Switch:1
+BackupVariables/Conv_out__/beta/cond/read:0
/BackupVariables/Conv_out__/beta/cond/switch_t:0
Conv_out__/beta:0`
.BackupVariables/Conv_out__/beta/cond/pred_id:0.BackupVariables/Conv_out__/beta/cond/pred_id:0G
Conv_out__/beta:02BackupVariables/Conv_out__/beta/cond/read/Switch:1
?
0BackupVariables/Conv_out__/beta/cond/cond_text_1.BackupVariables/Conv_out__/beta/cond/pred_id:0/BackupVariables/Conv_out__/beta/cond/switch_f:0*?
/BackupVariables/Conv_out__/beta/cond/Switch_1:0
/BackupVariables/Conv_out__/beta/cond/Switch_1:1
.BackupVariables/Conv_out__/beta/cond/pred_id:0
/BackupVariables/Conv_out__/beta/cond/switch_f:0
#Conv_out__/beta/Initializer/zeros:0V
#Conv_out__/beta/Initializer/zeros:0/BackupVariables/Conv_out__/beta/cond/Switch_1:0`
.BackupVariables/Conv_out__/beta/cond/pred_id:0.BackupVariables/Conv_out__/beta/cond/pred_id:0
?
!BackupVariables/cond_13/cond_text!BackupVariables/cond_13/pred_id:0"BackupVariables/cond_13/switch_t:0 *?
!BackupVariables/cond_13/pred_id:0
%BackupVariables/cond_13/read/Switch:1
BackupVariables/cond_13/read:0
"BackupVariables/cond_13/switch_t:0
Conv_out__/gamma:0F
!BackupVariables/cond_13/pred_id:0!BackupVariables/cond_13/pred_id:0;
Conv_out__/gamma:0%BackupVariables/cond_13/read/Switch:1
?
#BackupVariables/cond_13/cond_text_1!BackupVariables/cond_13/pred_id:0"BackupVariables/cond_13/switch_f:0*?
"BackupVariables/cond_13/Switch_1:0
"BackupVariables/cond_13/Switch_1:1
!BackupVariables/cond_13/pred_id:0
"BackupVariables/cond_13/switch_f:0
#Conv_out__/gamma/Initializer/ones:0I
#Conv_out__/gamma/Initializer/ones:0"BackupVariables/cond_13/Switch_1:0F
!BackupVariables/cond_13/pred_id:0!BackupVariables/cond_13/pred_id:0
?
/BackupVariables/Conv_out__/gamma/cond/cond_text/BackupVariables/Conv_out__/gamma/cond/pred_id:00BackupVariables/Conv_out__/gamma/cond/switch_t:0 *?
/BackupVariables/Conv_out__/gamma/cond/pred_id:0
3BackupVariables/Conv_out__/gamma/cond/read/Switch:1
,BackupVariables/Conv_out__/gamma/cond/read:0
0BackupVariables/Conv_out__/gamma/cond/switch_t:0
Conv_out__/gamma:0b
/BackupVariables/Conv_out__/gamma/cond/pred_id:0/BackupVariables/Conv_out__/gamma/cond/pred_id:0I
Conv_out__/gamma:03BackupVariables/Conv_out__/gamma/cond/read/Switch:1
?
1BackupVariables/Conv_out__/gamma/cond/cond_text_1/BackupVariables/Conv_out__/gamma/cond/pred_id:00BackupVariables/Conv_out__/gamma/cond/switch_f:0*?
0BackupVariables/Conv_out__/gamma/cond/Switch_1:0
0BackupVariables/Conv_out__/gamma/cond/Switch_1:1
/BackupVariables/Conv_out__/gamma/cond/pred_id:0
0BackupVariables/Conv_out__/gamma/cond/switch_f:0
#Conv_out__/gamma/Initializer/ones:0W
#Conv_out__/gamma/Initializer/ones:00BackupVariables/Conv_out__/gamma/cond/Switch_1:0b
/BackupVariables/Conv_out__/gamma/cond/pred_id:0/BackupVariables/Conv_out__/gamma/cond/pred_id:0
?
!BackupVariables/cond_14/cond_text!BackupVariables/cond_14/pred_id:0"BackupVariables/cond_14/switch_t:0 *?
!BackupVariables/cond_14/pred_id:0
%BackupVariables/cond_14/read/Switch:1
BackupVariables/cond_14/read:0
"BackupVariables/cond_14/switch_t:0
$Reconstruction_Output/dense/kernel:0M
$Reconstruction_Output/dense/kernel:0%BackupVariables/cond_14/read/Switch:1F
!BackupVariables/cond_14/pred_id:0!BackupVariables/cond_14/pred_id:0
?
#BackupVariables/cond_14/cond_text_1!BackupVariables/cond_14/pred_id:0"BackupVariables/cond_14/switch_f:0*?
"BackupVariables/cond_14/Switch_1:0
"BackupVariables/cond_14/Switch_1:1
!BackupVariables/cond_14/pred_id:0
"BackupVariables/cond_14/switch_f:0
?Reconstruction_Output/dense/kernel/Initializer/random_uniform:0e
?Reconstruction_Output/dense/kernel/Initializer/random_uniform:0"BackupVariables/cond_14/Switch_1:0F
!BackupVariables/cond_14/pred_id:0!BackupVariables/cond_14/pred_id:0
?
ABackupVariables/Reconstruction_Output/dense/kernel/cond/cond_textABackupVariables/Reconstruction_Output/dense/kernel/cond/pred_id:0BBackupVariables/Reconstruction_Output/dense/kernel/cond/switch_t:0 *?
ABackupVariables/Reconstruction_Output/dense/kernel/cond/pred_id:0
EBackupVariables/Reconstruction_Output/dense/kernel/cond/read/Switch:1
>BackupVariables/Reconstruction_Output/dense/kernel/cond/read:0
BBackupVariables/Reconstruction_Output/dense/kernel/cond/switch_t:0
$Reconstruction_Output/dense/kernel:0?
ABackupVariables/Reconstruction_Output/dense/kernel/cond/pred_id:0ABackupVariables/Reconstruction_Output/dense/kernel/cond/pred_id:0m
$Reconstruction_Output/dense/kernel:0EBackupVariables/Reconstruction_Output/dense/kernel/cond/read/Switch:1
?
CBackupVariables/Reconstruction_Output/dense/kernel/cond/cond_text_1ABackupVariables/Reconstruction_Output/dense/kernel/cond/pred_id:0BBackupVariables/Reconstruction_Output/dense/kernel/cond/switch_f:0*?
BBackupVariables/Reconstruction_Output/dense/kernel/cond/Switch_1:0
BBackupVariables/Reconstruction_Output/dense/kernel/cond/Switch_1:1
ABackupVariables/Reconstruction_Output/dense/kernel/cond/pred_id:0
BBackupVariables/Reconstruction_Output/dense/kernel/cond/switch_f:0
?Reconstruction_Output/dense/kernel/Initializer/random_uniform:0?
ABackupVariables/Reconstruction_Output/dense/kernel/cond/pred_id:0ABackupVariables/Reconstruction_Output/dense/kernel/cond/pred_id:0?
?Reconstruction_Output/dense/kernel/Initializer/random_uniform:0BBackupVariables/Reconstruction_Output/dense/kernel/cond/Switch_1:0
?
!BackupVariables/cond_15/cond_text!BackupVariables/cond_15/pred_id:0"BackupVariables/cond_15/switch_t:0 *?
!BackupVariables/cond_15/pred_id:0
%BackupVariables/cond_15/read/Switch:1
BackupVariables/cond_15/read:0
"BackupVariables/cond_15/switch_t:0
"Reconstruction_Output/dense/bias:0K
"Reconstruction_Output/dense/bias:0%BackupVariables/cond_15/read/Switch:1F
!BackupVariables/cond_15/pred_id:0!BackupVariables/cond_15/pred_id:0
?
#BackupVariables/cond_15/cond_text_1!BackupVariables/cond_15/pred_id:0"BackupVariables/cond_15/switch_f:0*?
"BackupVariables/cond_15/Switch_1:0
"BackupVariables/cond_15/Switch_1:1
!BackupVariables/cond_15/pred_id:0
"BackupVariables/cond_15/switch_f:0
4Reconstruction_Output/dense/bias/Initializer/zeros:0Z
4Reconstruction_Output/dense/bias/Initializer/zeros:0"BackupVariables/cond_15/Switch_1:0F
!BackupVariables/cond_15/pred_id:0!BackupVariables/cond_15/pred_id:0
?
?BackupVariables/Reconstruction_Output/dense/bias/cond/cond_text?BackupVariables/Reconstruction_Output/dense/bias/cond/pred_id:0@BackupVariables/Reconstruction_Output/dense/bias/cond/switch_t:0 *?
?BackupVariables/Reconstruction_Output/dense/bias/cond/pred_id:0
CBackupVariables/Reconstruction_Output/dense/bias/cond/read/Switch:1
<BackupVariables/Reconstruction_Output/dense/bias/cond/read:0
@BackupVariables/Reconstruction_Output/dense/bias/cond/switch_t:0
"Reconstruction_Output/dense/bias:0?
?BackupVariables/Reconstruction_Output/dense/bias/cond/pred_id:0?BackupVariables/Reconstruction_Output/dense/bias/cond/pred_id:0i
"Reconstruction_Output/dense/bias:0CBackupVariables/Reconstruction_Output/dense/bias/cond/read/Switch:1
?
ABackupVariables/Reconstruction_Output/dense/bias/cond/cond_text_1?BackupVariables/Reconstruction_Output/dense/bias/cond/pred_id:0@BackupVariables/Reconstruction_Output/dense/bias/cond/switch_f:0*?
@BackupVariables/Reconstruction_Output/dense/bias/cond/Switch_1:0
@BackupVariables/Reconstruction_Output/dense/bias/cond/Switch_1:1
?BackupVariables/Reconstruction_Output/dense/bias/cond/pred_id:0
@BackupVariables/Reconstruction_Output/dense/bias/cond/switch_f:0
4Reconstruction_Output/dense/bias/Initializer/zeros:0?
?BackupVariables/Reconstruction_Output/dense/bias/cond/pred_id:0?BackupVariables/Reconstruction_Output/dense/bias/cond/pred_id:0x
4Reconstruction_Output/dense/bias/Initializer/zeros:0@BackupVariables/Reconstruction_Output/dense/bias/cond/Switch_1:0
?
!BackupVariables/cond_16/cond_text!BackupVariables/cond_16/pred_id:0"BackupVariables/cond_16/switch_t:0 *?
!BackupVariables/cond_16/pred_id:0
%BackupVariables/cond_16/read/Switch:1
BackupVariables/cond_16/read:0
"BackupVariables/cond_16/switch_t:0
dense/kernel:07
dense/kernel:0%BackupVariables/cond_16/read/Switch:1F
!BackupVariables/cond_16/pred_id:0!BackupVariables/cond_16/pred_id:0
?
#BackupVariables/cond_16/cond_text_1!BackupVariables/cond_16/pred_id:0"BackupVariables/cond_16/switch_f:0*?
"BackupVariables/cond_16/Switch_1:0
"BackupVariables/cond_16/Switch_1:1
!BackupVariables/cond_16/pred_id:0
"BackupVariables/cond_16/switch_f:0
)dense/kernel/Initializer/random_uniform:0F
!BackupVariables/cond_16/pred_id:0!BackupVariables/cond_16/pred_id:0O
)dense/kernel/Initializer/random_uniform:0"BackupVariables/cond_16/Switch_1:0
?
+BackupVariables/dense/kernel/cond/cond_text+BackupVariables/dense/kernel/cond/pred_id:0,BackupVariables/dense/kernel/cond/switch_t:0 *?
+BackupVariables/dense/kernel/cond/pred_id:0
/BackupVariables/dense/kernel/cond/read/Switch:1
(BackupVariables/dense/kernel/cond/read:0
,BackupVariables/dense/kernel/cond/switch_t:0
dense/kernel:0Z
+BackupVariables/dense/kernel/cond/pred_id:0+BackupVariables/dense/kernel/cond/pred_id:0A
dense/kernel:0/BackupVariables/dense/kernel/cond/read/Switch:1
?
-BackupVariables/dense/kernel/cond/cond_text_1+BackupVariables/dense/kernel/cond/pred_id:0,BackupVariables/dense/kernel/cond/switch_f:0*?
,BackupVariables/dense/kernel/cond/Switch_1:0
,BackupVariables/dense/kernel/cond/Switch_1:1
+BackupVariables/dense/kernel/cond/pred_id:0
,BackupVariables/dense/kernel/cond/switch_f:0
)dense/kernel/Initializer/random_uniform:0Y
)dense/kernel/Initializer/random_uniform:0,BackupVariables/dense/kernel/cond/Switch_1:0Z
+BackupVariables/dense/kernel/cond/pred_id:0+BackupVariables/dense/kernel/cond/pred_id:0
?
!BackupVariables/cond_17/cond_text!BackupVariables/cond_17/pred_id:0"BackupVariables/cond_17/switch_t:0 *?
!BackupVariables/cond_17/pred_id:0
%BackupVariables/cond_17/read/Switch:1
BackupVariables/cond_17/read:0
"BackupVariables/cond_17/switch_t:0
dense/bias:05
dense/bias:0%BackupVariables/cond_17/read/Switch:1F
!BackupVariables/cond_17/pred_id:0!BackupVariables/cond_17/pred_id:0
?
#BackupVariables/cond_17/cond_text_1!BackupVariables/cond_17/pred_id:0"BackupVariables/cond_17/switch_f:0*?
"BackupVariables/cond_17/Switch_1:0
"BackupVariables/cond_17/Switch_1:1
!BackupVariables/cond_17/pred_id:0
"BackupVariables/cond_17/switch_f:0
dense/bias/Initializer/zeros:0D
dense/bias/Initializer/zeros:0"BackupVariables/cond_17/Switch_1:0F
!BackupVariables/cond_17/pred_id:0!BackupVariables/cond_17/pred_id:0
?
)BackupVariables/dense/bias/cond/cond_text)BackupVariables/dense/bias/cond/pred_id:0*BackupVariables/dense/bias/cond/switch_t:0 *?
)BackupVariables/dense/bias/cond/pred_id:0
-BackupVariables/dense/bias/cond/read/Switch:1
&BackupVariables/dense/bias/cond/read:0
*BackupVariables/dense/bias/cond/switch_t:0
dense/bias:0=
dense/bias:0-BackupVariables/dense/bias/cond/read/Switch:1V
)BackupVariables/dense/bias/cond/pred_id:0)BackupVariables/dense/bias/cond/pred_id:0
?
+BackupVariables/dense/bias/cond/cond_text_1)BackupVariables/dense/bias/cond/pred_id:0*BackupVariables/dense/bias/cond/switch_f:0*?
*BackupVariables/dense/bias/cond/Switch_1:0
*BackupVariables/dense/bias/cond/Switch_1:1
)BackupVariables/dense/bias/cond/pred_id:0
*BackupVariables/dense/bias/cond/switch_f:0
dense/bias/Initializer/zeros:0L
dense/bias/Initializer/zeros:0*BackupVariables/dense/bias/cond/Switch_1:0V
)BackupVariables/dense/bias/cond/pred_id:0)BackupVariables/dense/bias/cond/pred_id:0
?
!BackupVariables/cond_18/cond_text!BackupVariables/cond_18/pred_id:0"BackupVariables/cond_18/switch_t:0 *?
!BackupVariables/cond_18/pred_id:0
%BackupVariables/cond_18/read/Switch:1
BackupVariables/cond_18/read:0
"BackupVariables/cond_18/switch_t:0
 FCU_muiltDense_x0/dense/kernel:0I
 FCU_muiltDense_x0/dense/kernel:0%BackupVariables/cond_18/read/Switch:1F
!BackupVariables/cond_18/pred_id:0!BackupVariables/cond_18/pred_id:0
?
#BackupVariables/cond_18/cond_text_1!BackupVariables/cond_18/pred_id:0"BackupVariables/cond_18/switch_f:0*?
"BackupVariables/cond_18/Switch_1:0
"BackupVariables/cond_18/Switch_1:1
!BackupVariables/cond_18/pred_id:0
"BackupVariables/cond_18/switch_f:0
;FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform:0a
;FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform:0"BackupVariables/cond_18/Switch_1:0F
!BackupVariables/cond_18/pred_id:0!BackupVariables/cond_18/pred_id:0
?
=BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/cond_text=BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/pred_id:0>BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/switch_t:0 *?
=BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/pred_id:0
ABackupVariables/FCU_muiltDense_x0/dense/kernel/cond/read/Switch:1
:BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/read:0
>BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/switch_t:0
 FCU_muiltDense_x0/dense/kernel:0~
=BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/pred_id:0=BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/pred_id:0e
 FCU_muiltDense_x0/dense/kernel:0ABackupVariables/FCU_muiltDense_x0/dense/kernel/cond/read/Switch:1
?
?BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/cond_text_1=BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/pred_id:0>BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/switch_f:0*?
>BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/Switch_1:0
>BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/Switch_1:1
=BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/pred_id:0
>BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/switch_f:0
;FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform:0}
;FCU_muiltDense_x0/dense/kernel/Initializer/random_uniform:0>BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/Switch_1:0~
=BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/pred_id:0=BackupVariables/FCU_muiltDense_x0/dense/kernel/cond/pred_id:0
?
!BackupVariables/cond_19/cond_text!BackupVariables/cond_19/pred_id:0"BackupVariables/cond_19/switch_t:0 *?
!BackupVariables/cond_19/pred_id:0
%BackupVariables/cond_19/read/Switch:1
BackupVariables/cond_19/read:0
"BackupVariables/cond_19/switch_t:0
FCU_muiltDense_x0/dense/bias:0G
FCU_muiltDense_x0/dense/bias:0%BackupVariables/cond_19/read/Switch:1F
!BackupVariables/cond_19/pred_id:0!BackupVariables/cond_19/pred_id:0
?
#BackupVariables/cond_19/cond_text_1!BackupVariables/cond_19/pred_id:0"BackupVariables/cond_19/switch_f:0*?
"BackupVariables/cond_19/Switch_1:0
"BackupVariables/cond_19/Switch_1:1
!BackupVariables/cond_19/pred_id:0
"BackupVariables/cond_19/switch_f:0
0FCU_muiltDense_x0/dense/bias/Initializer/zeros:0V
0FCU_muiltDense_x0/dense/bias/Initializer/zeros:0"BackupVariables/cond_19/Switch_1:0F
!BackupVariables/cond_19/pred_id:0!BackupVariables/cond_19/pred_id:0
?
;BackupVariables/FCU_muiltDense_x0/dense/bias/cond/cond_text;BackupVariables/FCU_muiltDense_x0/dense/bias/cond/pred_id:0<BackupVariables/FCU_muiltDense_x0/dense/bias/cond/switch_t:0 *?
;BackupVariables/FCU_muiltDense_x0/dense/bias/cond/pred_id:0
?BackupVariables/FCU_muiltDense_x0/dense/bias/cond/read/Switch:1
8BackupVariables/FCU_muiltDense_x0/dense/bias/cond/read:0
<BackupVariables/FCU_muiltDense_x0/dense/bias/cond/switch_t:0
FCU_muiltDense_x0/dense/bias:0z
;BackupVariables/FCU_muiltDense_x0/dense/bias/cond/pred_id:0;BackupVariables/FCU_muiltDense_x0/dense/bias/cond/pred_id:0a
FCU_muiltDense_x0/dense/bias:0?BackupVariables/FCU_muiltDense_x0/dense/bias/cond/read/Switch:1
?
=BackupVariables/FCU_muiltDense_x0/dense/bias/cond/cond_text_1;BackupVariables/FCU_muiltDense_x0/dense/bias/cond/pred_id:0<BackupVariables/FCU_muiltDense_x0/dense/bias/cond/switch_f:0*?
<BackupVariables/FCU_muiltDense_x0/dense/bias/cond/Switch_1:0
<BackupVariables/FCU_muiltDense_x0/dense/bias/cond/Switch_1:1
;BackupVariables/FCU_muiltDense_x0/dense/bias/cond/pred_id:0
<BackupVariables/FCU_muiltDense_x0/dense/bias/cond/switch_f:0
0FCU_muiltDense_x0/dense/bias/Initializer/zeros:0p
0FCU_muiltDense_x0/dense/bias/Initializer/zeros:0<BackupVariables/FCU_muiltDense_x0/dense/bias/cond/Switch_1:0z
;BackupVariables/FCU_muiltDense_x0/dense/bias/cond/pred_id:0;BackupVariables/FCU_muiltDense_x0/dense/bias/cond/pred_id:0
?
!BackupVariables/cond_20/cond_text!BackupVariables/cond_20/pred_id:0"BackupVariables/cond_20/switch_t:0 *?
!BackupVariables/cond_20/pred_id:0
%BackupVariables/cond_20/read/Switch:1
BackupVariables/cond_20/read:0
"BackupVariables/cond_20/switch_t:0
FCU_muiltDense_x0/beta:0A
FCU_muiltDense_x0/beta:0%BackupVariables/cond_20/read/Switch:1F
!BackupVariables/cond_20/pred_id:0!BackupVariables/cond_20/pred_id:0
?
#BackupVariables/cond_20/cond_text_1!BackupVariables/cond_20/pred_id:0"BackupVariables/cond_20/switch_f:0*?
"BackupVariables/cond_20/Switch_1:0
"BackupVariables/cond_20/Switch_1:1
!BackupVariables/cond_20/pred_id:0
"BackupVariables/cond_20/switch_f:0
*FCU_muiltDense_x0/beta/Initializer/zeros:0F
!BackupVariables/cond_20/pred_id:0!BackupVariables/cond_20/pred_id:0P
*FCU_muiltDense_x0/beta/Initializer/zeros:0"BackupVariables/cond_20/Switch_1:0
?
5BackupVariables/FCU_muiltDense_x0/beta/cond/cond_text5BackupVariables/FCU_muiltDense_x0/beta/cond/pred_id:06BackupVariables/FCU_muiltDense_x0/beta/cond/switch_t:0 *?
5BackupVariables/FCU_muiltDense_x0/beta/cond/pred_id:0
9BackupVariables/FCU_muiltDense_x0/beta/cond/read/Switch:1
2BackupVariables/FCU_muiltDense_x0/beta/cond/read:0
6BackupVariables/FCU_muiltDense_x0/beta/cond/switch_t:0
FCU_muiltDense_x0/beta:0U
FCU_muiltDense_x0/beta:09BackupVariables/FCU_muiltDense_x0/beta/cond/read/Switch:1n
5BackupVariables/FCU_muiltDense_x0/beta/cond/pred_id:05BackupVariables/FCU_muiltDense_x0/beta/cond/pred_id:0
?
7BackupVariables/FCU_muiltDense_x0/beta/cond/cond_text_15BackupVariables/FCU_muiltDense_x0/beta/cond/pred_id:06BackupVariables/FCU_muiltDense_x0/beta/cond/switch_f:0*?
6BackupVariables/FCU_muiltDense_x0/beta/cond/Switch_1:0
6BackupVariables/FCU_muiltDense_x0/beta/cond/Switch_1:1
5BackupVariables/FCU_muiltDense_x0/beta/cond/pred_id:0
6BackupVariables/FCU_muiltDense_x0/beta/cond/switch_f:0
*FCU_muiltDense_x0/beta/Initializer/zeros:0d
*FCU_muiltDense_x0/beta/Initializer/zeros:06BackupVariables/FCU_muiltDense_x0/beta/cond/Switch_1:0n
5BackupVariables/FCU_muiltDense_x0/beta/cond/pred_id:05BackupVariables/FCU_muiltDense_x0/beta/cond/pred_id:0
?
!BackupVariables/cond_21/cond_text!BackupVariables/cond_21/pred_id:0"BackupVariables/cond_21/switch_t:0 *?
!BackupVariables/cond_21/pred_id:0
%BackupVariables/cond_21/read/Switch:1
BackupVariables/cond_21/read:0
"BackupVariables/cond_21/switch_t:0
FCU_muiltDense_x0/gamma:0F
!BackupVariables/cond_21/pred_id:0!BackupVariables/cond_21/pred_id:0B
FCU_muiltDense_x0/gamma:0%BackupVariables/cond_21/read/Switch:1
?
#BackupVariables/cond_21/cond_text_1!BackupVariables/cond_21/pred_id:0"BackupVariables/cond_21/switch_f:0*?
"BackupVariables/cond_21/Switch_1:0
"BackupVariables/cond_21/Switch_1:1
!BackupVariables/cond_21/pred_id:0
"BackupVariables/cond_21/switch_f:0
*FCU_muiltDense_x0/gamma/Initializer/ones:0P
*FCU_muiltDense_x0/gamma/Initializer/ones:0"BackupVariables/cond_21/Switch_1:0F
!BackupVariables/cond_21/pred_id:0!BackupVariables/cond_21/pred_id:0
?
6BackupVariables/FCU_muiltDense_x0/gamma/cond/cond_text6BackupVariables/FCU_muiltDense_x0/gamma/cond/pred_id:07BackupVariables/FCU_muiltDense_x0/gamma/cond/switch_t:0 *?
6BackupVariables/FCU_muiltDense_x0/gamma/cond/pred_id:0
:BackupVariables/FCU_muiltDense_x0/gamma/cond/read/Switch:1
3BackupVariables/FCU_muiltDense_x0/gamma/cond/read:0
7BackupVariables/FCU_muiltDense_x0/gamma/cond/switch_t:0
FCU_muiltDense_x0/gamma:0p
6BackupVariables/FCU_muiltDense_x0/gamma/cond/pred_id:06BackupVariables/FCU_muiltDense_x0/gamma/cond/pred_id:0W
FCU_muiltDense_x0/gamma:0:BackupVariables/FCU_muiltDense_x0/gamma/cond/read/Switch:1
?
8BackupVariables/FCU_muiltDense_x0/gamma/cond/cond_text_16BackupVariables/FCU_muiltDense_x0/gamma/cond/pred_id:07BackupVariables/FCU_muiltDense_x0/gamma/cond/switch_f:0*?
7BackupVariables/FCU_muiltDense_x0/gamma/cond/Switch_1:0
7BackupVariables/FCU_muiltDense_x0/gamma/cond/Switch_1:1
6BackupVariables/FCU_muiltDense_x0/gamma/cond/pred_id:0
7BackupVariables/FCU_muiltDense_x0/gamma/cond/switch_f:0
*FCU_muiltDense_x0/gamma/Initializer/ones:0p
6BackupVariables/FCU_muiltDense_x0/gamma/cond/pred_id:06BackupVariables/FCU_muiltDense_x0/gamma/cond/pred_id:0e
*FCU_muiltDense_x0/gamma/Initializer/ones:07BackupVariables/FCU_muiltDense_x0/gamma/cond/Switch_1:0
?
!BackupVariables/cond_22/cond_text!BackupVariables/cond_22/pred_id:0"BackupVariables/cond_22/switch_t:0 *?
!BackupVariables/cond_22/pred_id:0
%BackupVariables/cond_22/read/Switch:1
BackupVariables/cond_22/read:0
"BackupVariables/cond_22/switch_t:0
Output_/dense/kernel:0?
Output_/dense/kernel:0%BackupVariables/cond_22/read/Switch:1F
!BackupVariables/cond_22/pred_id:0!BackupVariables/cond_22/pred_id:0
?
#BackupVariables/cond_22/cond_text_1!BackupVariables/cond_22/pred_id:0"BackupVariables/cond_22/switch_f:0*?
"BackupVariables/cond_22/Switch_1:0
"BackupVariables/cond_22/Switch_1:1
!BackupVariables/cond_22/pred_id:0
"BackupVariables/cond_22/switch_f:0
1Output_/dense/kernel/Initializer/random_uniform:0W
1Output_/dense/kernel/Initializer/random_uniform:0"BackupVariables/cond_22/Switch_1:0F
!BackupVariables/cond_22/pred_id:0!BackupVariables/cond_22/pred_id:0
?
3BackupVariables/Output_/dense/kernel/cond/cond_text3BackupVariables/Output_/dense/kernel/cond/pred_id:04BackupVariables/Output_/dense/kernel/cond/switch_t:0 *?
3BackupVariables/Output_/dense/kernel/cond/pred_id:0
7BackupVariables/Output_/dense/kernel/cond/read/Switch:1
0BackupVariables/Output_/dense/kernel/cond/read:0
4BackupVariables/Output_/dense/kernel/cond/switch_t:0
Output_/dense/kernel:0Q
Output_/dense/kernel:07BackupVariables/Output_/dense/kernel/cond/read/Switch:1j
3BackupVariables/Output_/dense/kernel/cond/pred_id:03BackupVariables/Output_/dense/kernel/cond/pred_id:0
?
5BackupVariables/Output_/dense/kernel/cond/cond_text_13BackupVariables/Output_/dense/kernel/cond/pred_id:04BackupVariables/Output_/dense/kernel/cond/switch_f:0*?
4BackupVariables/Output_/dense/kernel/cond/Switch_1:0
4BackupVariables/Output_/dense/kernel/cond/Switch_1:1
3BackupVariables/Output_/dense/kernel/cond/pred_id:0
4BackupVariables/Output_/dense/kernel/cond/switch_f:0
1Output_/dense/kernel/Initializer/random_uniform:0j
3BackupVariables/Output_/dense/kernel/cond/pred_id:03BackupVariables/Output_/dense/kernel/cond/pred_id:0i
1Output_/dense/kernel/Initializer/random_uniform:04BackupVariables/Output_/dense/kernel/cond/Switch_1:0
?
!BackupVariables/cond_23/cond_text!BackupVariables/cond_23/pred_id:0"BackupVariables/cond_23/switch_t:0 *?
!BackupVariables/cond_23/pred_id:0
%BackupVariables/cond_23/read/Switch:1
BackupVariables/cond_23/read:0
"BackupVariables/cond_23/switch_t:0
Output_/dense/bias:0F
!BackupVariables/cond_23/pred_id:0!BackupVariables/cond_23/pred_id:0=
Output_/dense/bias:0%BackupVariables/cond_23/read/Switch:1
?
#BackupVariables/cond_23/cond_text_1!BackupVariables/cond_23/pred_id:0"BackupVariables/cond_23/switch_f:0*?
"BackupVariables/cond_23/Switch_1:0
"BackupVariables/cond_23/Switch_1:1
!BackupVariables/cond_23/pred_id:0
"BackupVariables/cond_23/switch_f:0
&Output_/dense/bias/Initializer/zeros:0L
&Output_/dense/bias/Initializer/zeros:0"BackupVariables/cond_23/Switch_1:0F
!BackupVariables/cond_23/pred_id:0!BackupVariables/cond_23/pred_id:0
?
1BackupVariables/Output_/dense/bias/cond/cond_text1BackupVariables/Output_/dense/bias/cond/pred_id:02BackupVariables/Output_/dense/bias/cond/switch_t:0 *?
1BackupVariables/Output_/dense/bias/cond/pred_id:0
5BackupVariables/Output_/dense/bias/cond/read/Switch:1
.BackupVariables/Output_/dense/bias/cond/read:0
2BackupVariables/Output_/dense/bias/cond/switch_t:0
Output_/dense/bias:0M
Output_/dense/bias:05BackupVariables/Output_/dense/bias/cond/read/Switch:1f
1BackupVariables/Output_/dense/bias/cond/pred_id:01BackupVariables/Output_/dense/bias/cond/pred_id:0
?
3BackupVariables/Output_/dense/bias/cond/cond_text_11BackupVariables/Output_/dense/bias/cond/pred_id:02BackupVariables/Output_/dense/bias/cond/switch_f:0*?
2BackupVariables/Output_/dense/bias/cond/Switch_1:0
2BackupVariables/Output_/dense/bias/cond/Switch_1:1
1BackupVariables/Output_/dense/bias/cond/pred_id:0
2BackupVariables/Output_/dense/bias/cond/switch_f:0
&Output_/dense/bias/Initializer/zeros:0\
&Output_/dense/bias/Initializer/zeros:02BackupVariables/Output_/dense/bias/cond/Switch_1:0f
1BackupVariables/Output_/dense/bias/cond/pred_id:01BackupVariables/Output_/dense/bias/cond/pred_id:0"C
losses9
7
#My_GPU_1/mean_squared_error/value:0
My_GPU_1/add_3:0