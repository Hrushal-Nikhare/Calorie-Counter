¼·%
¦ü
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.22v2.8.2-0-g2ea19cbb5758Ü!
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel

'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0

block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel

'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0

block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock2_conv1/kernel

'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:*
dtype0

block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock2_conv2/kernel

'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv1/kernel

'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:*
dtype0

block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv2/kernel

'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:*
dtype0

block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv3/kernel

'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:*
dtype0

block3_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock3_conv4/kernel

'block3_conv4/kernel/Read/ReadVariableOpReadVariableOpblock3_conv4/kernel*(
_output_shapes
:*
dtype0
{
block3_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock3_conv4/bias
t
%block3_conv4/bias/Read/ReadVariableOpReadVariableOpblock3_conv4/bias*
_output_shapes	
:*
dtype0

block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv1/kernel

'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:*
dtype0

block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv2/kernel

'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:*
dtype0

block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv3/kernel

'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:*
dtype0

block4_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock4_conv4/kernel

'block4_conv4/kernel/Read/ReadVariableOpReadVariableOpblock4_conv4/kernel*(
_output_shapes
:*
dtype0
{
block4_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock4_conv4/bias
t
%block4_conv4/bias/Read/ReadVariableOpReadVariableOpblock4_conv4/bias*
_output_shapes	
:*
dtype0

block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv1/kernel

'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:*
dtype0

block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv2/kernel

'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:*
dtype0

block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv3/kernel

'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:*
dtype0

block5_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock5_conv4/kernel

'block5_conv4/kernel/Read/ReadVariableOpReadVariableOpblock5_conv4/kernel*(
_output_shapes
:*
dtype0
{
block5_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock5_conv4/bias
t
%block5_conv4/bias/Read/ReadVariableOpReadVariableOpblock5_conv4/bias*
_output_shapes	
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	@*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	@*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
°
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*É¯
value¾¯Bº¯ B²¯
Á
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
ø
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*

'iter

(beta_1

)beta_2
	*decay
+learning_ratem mLmMmv vLvMv*

,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31
L32
M33
34
 35*
 
L0
M1
2
 3*
* 
°
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Sserving_default* 
* 
à
Tlayer-0
Ulayer_with_weights-0
Ulayer-1
Vlayer_with_weights-1
Vlayer-2
Wlayer-3
Xlayer_with_weights-2
Xlayer-4
Ylayer_with_weights-3
Ylayer-5
Zlayer-6
[layer_with_weights-4
[layer-7
\layer_with_weights-5
\layer-8
]layer_with_weights-6
]layer-9
^layer_with_weights-7
^layer-10
_layer-11
`layer_with_weights-8
`layer-12
alayer_with_weights-9
alayer-13
blayer_with_weights-10
blayer-14
clayer_with_weights-11
clayer-15
dlayer-16
elayer_with_weights-12
elayer-17
flayer_with_weights-13
flayer-18
glayer_with_weights-14
glayer-19
hlayer_with_weights-15
hlayer-20
ilayer-21
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses*

p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 
¦

Lkernel
Mbias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses*

,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31
L32
M33*

L0
M1*
* 

|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv4/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv1/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv1/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv2/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv2/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv3/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv3/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv4/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv4/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv1/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv1/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv2/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv2/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv3/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv3/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv4/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv4/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_1/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
ú
,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31*
'
0
1
2
3
4*

0
1*
* 
* 
* 
* 
¬

,kernel
-bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬

.kernel
/bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬

0kernel
1bias
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£__call__
+¤&call_and_return_all_conditional_losses*
¬

2kernel
3bias
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
©__call__
+ª&call_and_return_all_conditional_losses*

«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯__call__
+°&call_and_return_all_conditional_losses* 
¬

4kernel
5bias
±	variables
²trainable_variables
³regularization_losses
´	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses*
¬

6kernel
7bias
·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
»__call__
+¼&call_and_return_all_conditional_losses*
¬

8kernel
9bias
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses*
¬

:kernel
;bias
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses*

É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses* 
¬

<kernel
=bias
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses*
¬

>kernel
?bias
Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses*
¬

@kernel
Abias
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses*
¬

Bkernel
Cbias
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
å__call__
+æ&call_and_return_all_conditional_losses*

ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses* 
¬

Dkernel
Ebias
í	variables
îtrainable_variables
ïregularization_losses
ð	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses*
¬

Fkernel
Gbias
ó	variables
ôtrainable_variables
õregularization_losses
ö	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses*
¬

Hkernel
Ibias
ù	variables
útrainable_variables
ûregularization_losses
ü	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses*
¬

Jkernel
Kbias
ÿ	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
ú
,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 
* 
* 

L0
M1*

L0
M1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*
* 
* 
ú
,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31*
 
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
	variables
	keras_api*
M

total

count
 
_fn_kwargs
¡	variables
¢	keras_api*

,0
-1*
* 
* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 

.0
/1*
* 
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

00
11*
* 
* 

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
	variables
 trainable_variables
¡regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses*
* 
* 

20
31*
* 
* 

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
¥	variables
¦trainable_variables
§regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
«	variables
¬trainable_variables
­regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses* 
* 
* 

40
51*
* 
* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
±	variables
²trainable_variables
³regularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses*
* 
* 

60
71*
* 
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
·	variables
¸trainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses*
* 
* 

80
91*
* 
* 

Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses*
* 
* 

:0
;1*
* 
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
É	variables
Êtrainable_variables
Ëregularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses* 
* 
* 

<0
=1*
* 
* 

Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses*
* 
* 

>0
?1*
* 
* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses*
* 
* 

@0
A1*
* 
* 

änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses*
* 
* 

B0
C1*
* 
* 

énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
á	variables
âtrainable_variables
ãregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses* 
* 
* 

D0
E1*
* 
* 

ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
í	variables
îtrainable_variables
ïregularization_losses
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses*
* 
* 

F0
G1*
* 
* 

ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
ó	variables
ôtrainable_variables
õregularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses*
* 
* 

H0
I1*
* 
* 

ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
ù	variables
útrainable_variables
ûregularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses*
* 
* 

J0
K1*
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ÿ	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
ú
,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31*
ª
T0
U1
V2
W3
X4
Y5
Z6
[7
\8
]9
^10
_11
`12
a13
b14
c15
d16
e17
f18
g19
h20
i21*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

¡	variables*

,0
-1*
* 
* 
* 
* 

.0
/1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

00
11*
* 
* 
* 
* 

20
31*
* 
* 
* 
* 
* 
* 
* 
* 
* 

40
51*
* 
* 
* 
* 

60
71*
* 
* 
* 
* 

80
91*
* 
* 
* 
* 

:0
;1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

<0
=1*
* 
* 
* 
* 

>0
?1*
* 
* 
* 
* 

@0
A1*
* 
* 
* 
* 

B0
C1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

D0
E1*
* 
* 
* 
* 

F0
G1*
* 
* 
* 
* 

H0
I1*
* 
* 
* 
* 

J0
K1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_3Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿÈÈ

serving_default_input_4Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿÈÈ
Ô
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3serving_default_input_4block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1037703
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
É
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block3_conv4/kernel/Read/ReadVariableOp%block3_conv4/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block4_conv4/kernel/Read/ReadVariableOp%block4_conv4/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp'block5_conv4/kernel/Read/ReadVariableOp%block5_conv4/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*B
Tin;
927	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1039130
 

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock3_conv4/kernelblock3_conv4/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock4_conv4/kernelblock4_conv4/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasblock5_conv4/kernelblock5_conv4/biasdense_1/kerneldense_1/biastotalcounttotal_1count_1Adam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*A
Tin:
826*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1039299Õ
þ
¦
.__inference_block5_conv4_layer_call_fn_1038926

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv4_layer_call_and_return_conditional_losses_1034812x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ùg
º
B__inference_vgg19_layer_call_and_return_conditional_losses_1035207

inputs.
block1_conv1_1035121:@"
block1_conv1_1035123:@.
block1_conv2_1035126:@@"
block1_conv2_1035128:@/
block2_conv1_1035132:@#
block2_conv1_1035134:	0
block2_conv2_1035137:#
block2_conv2_1035139:	0
block3_conv1_1035143:#
block3_conv1_1035145:	0
block3_conv2_1035148:#
block3_conv2_1035150:	0
block3_conv3_1035153:#
block3_conv3_1035155:	0
block3_conv4_1035158:#
block3_conv4_1035160:	0
block4_conv1_1035164:#
block4_conv1_1035166:	0
block4_conv2_1035169:#
block4_conv2_1035171:	0
block4_conv3_1035174:#
block4_conv3_1035176:	0
block4_conv4_1035179:#
block4_conv4_1035181:	0
block5_conv1_1035185:#
block5_conv1_1035187:	0
block5_conv2_1035190:#
block5_conv2_1035192:	0
block5_conv3_1035195:#
block5_conv3_1035197:	0
block5_conv4_1035200:#
block5_conv4_1035202:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_1035121block1_conv1_1035123*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block1_conv1_layer_call_and_return_conditional_losses_1034553·
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_1035126block1_conv2_1035128*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block1_conv2_layer_call_and_return_conditional_losses_1034570ñ
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_pool_layer_call_and_return_conditional_losses_1034484­
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_1035132block2_conv1_1035134*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block2_conv1_layer_call_and_return_conditional_losses_1034588¶
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_1035137block2_conv2_1035139*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block2_conv2_layer_call_and_return_conditional_losses_1034605ò
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_pool_layer_call_and_return_conditional_losses_1034496­
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_1035143block3_conv1_1035145*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv1_layer_call_and_return_conditional_losses_1034623¶
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_1035148block3_conv2_1035150*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv2_layer_call_and_return_conditional_losses_1034640¶
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_1035153block3_conv3_1035155*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv3_layer_call_and_return_conditional_losses_1034657¶
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_1035158block3_conv4_1035160*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv4_layer_call_and_return_conditional_losses_1034674ò
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_pool_layer_call_and_return_conditional_losses_1034508­
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_1035164block4_conv1_1035166*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv1_layer_call_and_return_conditional_losses_1034692¶
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_1035169block4_conv2_1035171*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv2_layer_call_and_return_conditional_losses_1034709¶
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_1035174block4_conv3_1035176*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv3_layer_call_and_return_conditional_losses_1034726¶
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_1035179block4_conv4_1035181*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv4_layer_call_and_return_conditional_losses_1034743ò
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_pool_layer_call_and_return_conditional_losses_1034520­
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_1035185block5_conv1_1035187*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv1_layer_call_and_return_conditional_losses_1034761¶
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_1035190block5_conv2_1035192*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv2_layer_call_and_return_conditional_losses_1034778¶
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_1035195block5_conv3_1035197*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv3_layer_call_and_return_conditional_losses_1034795¶
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_1035200block5_conv4_1035202*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv4_layer_call_and_return_conditional_losses_1034812ò
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_pool_layer_call_and_return_conditional_losses_1034532|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
ÿ
£
.__inference_block1_conv2_layer_call_fn_1038606

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block1_conv2_layer_call_and_return_conditional_losses_1034570y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
 
_user_specified_nameinputs

d
H__inference_block1_pool_layer_call_and_return_conditional_losses_1034484

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
T
(__inference_lambda_layer_call_fn_1038113
inputs_0
inputs_1
identity¾
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_1036276`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1


o
C__inference_lambda_layer_call_and_return_conditional_losses_1038147
inputs_0
inputs_1
identityP
subSubinputs_0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
SquareSquaresub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1
þ
¦
.__inference_block2_conv2_layer_call_fn_1038656

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block2_conv2_layer_call_and_return_conditional_losses_1034605x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿdd: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
³
ò

D__inference_model_1_layer_call_and_return_conditional_losses_1036149
input_5'
vgg19_1036077:@
vgg19_1036079:@'
vgg19_1036081:@@
vgg19_1036083:@(
vgg19_1036085:@
vgg19_1036087:	)
vgg19_1036089:
vgg19_1036091:	)
vgg19_1036093:
vgg19_1036095:	)
vgg19_1036097:
vgg19_1036099:	)
vgg19_1036101:
vgg19_1036103:	)
vgg19_1036105:
vgg19_1036107:	)
vgg19_1036109:
vgg19_1036111:	)
vgg19_1036113:
vgg19_1036115:	)
vgg19_1036117:
vgg19_1036119:	)
vgg19_1036121:
vgg19_1036123:	)
vgg19_1036125:
vgg19_1036127:	)
vgg19_1036129:
vgg19_1036131:	)
vgg19_1036133:
vgg19_1036135:	)
vgg19_1036137:
vgg19_1036139:	"
dense_1_1036143:	@
dense_1_1036145:@
identity¢dense_1/StatefulPartitionedCall¢vgg19/StatefulPartitionedCallò
vgg19/StatefulPartitionedCallStatefulPartitionedCallinput_5vgg19_1036077vgg19_1036079vgg19_1036081vgg19_1036083vgg19_1036085vgg19_1036087vgg19_1036089vgg19_1036091vgg19_1036093vgg19_1036095vgg19_1036097vgg19_1036099vgg19_1036101vgg19_1036103vgg19_1036105vgg19_1036107vgg19_1036109vgg19_1036111vgg19_1036113vgg19_1036115vgg19_1036117vgg19_1036119vgg19_1036121vgg19_1036123vgg19_1036125vgg19_1036127vgg19_1036129vgg19_1036131vgg19_1036133vgg19_1036135vgg19_1036137vgg19_1036139*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_vgg19_layer_call_and_return_conditional_losses_1035207
*global_average_pooling2d_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_1035531
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0dense_1_1036143dense_1_1036145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1035617w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp ^dense_1/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_5

d
H__inference_block4_pool_layer_call_and_return_conditional_losses_1034520

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block3_conv3_layer_call_and_return_conditional_losses_1038737

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs


m
C__inference_lambda_layer_call_and_return_conditional_losses_1036405

inputs
inputs_1
identityN
subSubinputsinputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
SquareSquaresub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
þ
¦
.__inference_block3_conv2_layer_call_fn_1038706

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv2_layer_call_and_return_conditional_losses_1034640x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs

d
H__inference_block5_pool_layer_call_and_return_conditional_losses_1038947

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


õ
D__inference_dense_2_layer_call_and_return_conditional_losses_1038167

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
ÿ	
)__inference_model_2_layer_call_fn_1037071
inputs_0
inputs_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_1036296o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/1
¿

D__inference_model_1_layer_call_and_return_conditional_losses_1037978

inputsK
1vgg19_block1_conv1_conv2d_readvariableop_resource:@@
2vgg19_block1_conv1_biasadd_readvariableop_resource:@K
1vgg19_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg19_block1_conv2_biasadd_readvariableop_resource:@L
1vgg19_block2_conv1_conv2d_readvariableop_resource:@A
2vgg19_block2_conv1_biasadd_readvariableop_resource:	M
1vgg19_block2_conv2_conv2d_readvariableop_resource:A
2vgg19_block2_conv2_biasadd_readvariableop_resource:	M
1vgg19_block3_conv1_conv2d_readvariableop_resource:A
2vgg19_block3_conv1_biasadd_readvariableop_resource:	M
1vgg19_block3_conv2_conv2d_readvariableop_resource:A
2vgg19_block3_conv2_biasadd_readvariableop_resource:	M
1vgg19_block3_conv3_conv2d_readvariableop_resource:A
2vgg19_block3_conv3_biasadd_readvariableop_resource:	M
1vgg19_block3_conv4_conv2d_readvariableop_resource:A
2vgg19_block3_conv4_biasadd_readvariableop_resource:	M
1vgg19_block4_conv1_conv2d_readvariableop_resource:A
2vgg19_block4_conv1_biasadd_readvariableop_resource:	M
1vgg19_block4_conv2_conv2d_readvariableop_resource:A
2vgg19_block4_conv2_biasadd_readvariableop_resource:	M
1vgg19_block4_conv3_conv2d_readvariableop_resource:A
2vgg19_block4_conv3_biasadd_readvariableop_resource:	M
1vgg19_block4_conv4_conv2d_readvariableop_resource:A
2vgg19_block4_conv4_biasadd_readvariableop_resource:	M
1vgg19_block5_conv1_conv2d_readvariableop_resource:A
2vgg19_block5_conv1_biasadd_readvariableop_resource:	M
1vgg19_block5_conv2_conv2d_readvariableop_resource:A
2vgg19_block5_conv2_biasadd_readvariableop_resource:	M
1vgg19_block5_conv3_conv2d_readvariableop_resource:A
2vgg19_block5_conv3_biasadd_readvariableop_resource:	M
1vgg19_block5_conv4_conv2d_readvariableop_resource:A
2vgg19_block5_conv4_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	@5
'dense_1_biasadd_readvariableop_resource:@
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢)vgg19/block1_conv1/BiasAdd/ReadVariableOp¢(vgg19/block1_conv1/Conv2D/ReadVariableOp¢)vgg19/block1_conv2/BiasAdd/ReadVariableOp¢(vgg19/block1_conv2/Conv2D/ReadVariableOp¢)vgg19/block2_conv1/BiasAdd/ReadVariableOp¢(vgg19/block2_conv1/Conv2D/ReadVariableOp¢)vgg19/block2_conv2/BiasAdd/ReadVariableOp¢(vgg19/block2_conv2/Conv2D/ReadVariableOp¢)vgg19/block3_conv1/BiasAdd/ReadVariableOp¢(vgg19/block3_conv1/Conv2D/ReadVariableOp¢)vgg19/block3_conv2/BiasAdd/ReadVariableOp¢(vgg19/block3_conv2/Conv2D/ReadVariableOp¢)vgg19/block3_conv3/BiasAdd/ReadVariableOp¢(vgg19/block3_conv3/Conv2D/ReadVariableOp¢)vgg19/block3_conv4/BiasAdd/ReadVariableOp¢(vgg19/block3_conv4/Conv2D/ReadVariableOp¢)vgg19/block4_conv1/BiasAdd/ReadVariableOp¢(vgg19/block4_conv1/Conv2D/ReadVariableOp¢)vgg19/block4_conv2/BiasAdd/ReadVariableOp¢(vgg19/block4_conv2/Conv2D/ReadVariableOp¢)vgg19/block4_conv3/BiasAdd/ReadVariableOp¢(vgg19/block4_conv3/Conv2D/ReadVariableOp¢)vgg19/block4_conv4/BiasAdd/ReadVariableOp¢(vgg19/block4_conv4/Conv2D/ReadVariableOp¢)vgg19/block5_conv1/BiasAdd/ReadVariableOp¢(vgg19/block5_conv1/Conv2D/ReadVariableOp¢)vgg19/block5_conv2/BiasAdd/ReadVariableOp¢(vgg19/block5_conv2/Conv2D/ReadVariableOp¢)vgg19/block5_conv3/BiasAdd/ReadVariableOp¢(vgg19/block5_conv3/Conv2D/ReadVariableOp¢)vgg19/block5_conv4/BiasAdd/ReadVariableOp¢(vgg19/block5_conv4/Conv2D/ReadVariableOp¢
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Á
vgg19/block1_conv1/Conv2DConv2Dinputs0vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¢
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0à
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¸
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
£
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¤
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¹
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
¤
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¹
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      º
global_average_pooling2d_1/MeanMean"vgg19/block5_pool/MaxPool:output:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_1/MatMulMatMul(global_average_pooling2d_1/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@÷
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
ô
¸	
)__inference_model_1_layer_call_fn_1035695
input_5!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035624o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_5


I__inference_block4_conv1_layer_call_and_return_conditional_losses_1038787

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
üg
»
B__inference_vgg19_layer_call_and_return_conditional_losses_1035432
input_6.
block1_conv1_1035346:@"
block1_conv1_1035348:@.
block1_conv2_1035351:@@"
block1_conv2_1035353:@/
block2_conv1_1035357:@#
block2_conv1_1035359:	0
block2_conv2_1035362:#
block2_conv2_1035364:	0
block3_conv1_1035368:#
block3_conv1_1035370:	0
block3_conv2_1035373:#
block3_conv2_1035375:	0
block3_conv3_1035378:#
block3_conv3_1035380:	0
block3_conv4_1035383:#
block3_conv4_1035385:	0
block4_conv1_1035389:#
block4_conv1_1035391:	0
block4_conv2_1035394:#
block4_conv2_1035396:	0
block4_conv3_1035399:#
block4_conv3_1035401:	0
block4_conv4_1035404:#
block4_conv4_1035406:	0
block5_conv1_1035410:#
block5_conv1_1035412:	0
block5_conv2_1035415:#
block5_conv2_1035417:	0
block5_conv3_1035420:#
block5_conv3_1035422:	0
block5_conv4_1035425:#
block5_conv4_1035427:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_6block1_conv1_1035346block1_conv1_1035348*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block1_conv1_layer_call_and_return_conditional_losses_1034553·
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_1035351block1_conv2_1035353*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block1_conv2_layer_call_and_return_conditional_losses_1034570ñ
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_pool_layer_call_and_return_conditional_losses_1034484­
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_1035357block2_conv1_1035359*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block2_conv1_layer_call_and_return_conditional_losses_1034588¶
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_1035362block2_conv2_1035364*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block2_conv2_layer_call_and_return_conditional_losses_1034605ò
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_pool_layer_call_and_return_conditional_losses_1034496­
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_1035368block3_conv1_1035370*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv1_layer_call_and_return_conditional_losses_1034623¶
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_1035373block3_conv2_1035375*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv2_layer_call_and_return_conditional_losses_1034640¶
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_1035378block3_conv3_1035380*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv3_layer_call_and_return_conditional_losses_1034657¶
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_1035383block3_conv4_1035385*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv4_layer_call_and_return_conditional_losses_1034674ò
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_pool_layer_call_and_return_conditional_losses_1034508­
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_1035389block4_conv1_1035391*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv1_layer_call_and_return_conditional_losses_1034692¶
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_1035394block4_conv2_1035396*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv2_layer_call_and_return_conditional_losses_1034709¶
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_1035399block4_conv3_1035401*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv3_layer_call_and_return_conditional_losses_1034726¶
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_1035404block4_conv4_1035406*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv4_layer_call_and_return_conditional_losses_1034743ò
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_pool_layer_call_and_return_conditional_losses_1034520­
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_1035410block5_conv1_1035412*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv1_layer_call_and_return_conditional_losses_1034761¶
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_1035415block5_conv2_1035417*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv2_layer_call_and_return_conditional_losses_1034778¶
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_1035420block5_conv3_1035422*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv3_layer_call_and_return_conditional_losses_1034795¶
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_1035425block5_conv4_1035427*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv4_layer_call_and_return_conditional_losses_1034812ò
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_pool_layer_call_and_return_conditional_losses_1034532|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_6


I__inference_block4_conv4_layer_call_and_return_conditional_losses_1034743

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block1_conv1_layer_call_and_return_conditional_losses_1038597

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs

X
<__inference_global_average_pooling2d_1_layer_call_fn_1038552

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_1035531i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
ê
B__inference_vgg19_layer_call_and_return_conditional_losses_1038426

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	G
+block4_conv2_conv2d_readvariableop_resource:;
,block4_conv2_biasadd_readvariableop_resource:	G
+block4_conv3_conv2d_readvariableop_resource:;
,block4_conv3_biasadd_readvariableop_resource:	G
+block4_conv4_conv2d_readvariableop_resource:;
,block4_conv4_biasadd_readvariableop_resource:	G
+block5_conv1_conv2d_readvariableop_resource:;
,block5_conv1_biasadd_readvariableop_resource:	G
+block5_conv2_conv2d_readvariableop_resource:;
,block5_conv2_biasadd_readvariableop_resource:	G
+block5_conv3_conv2d_readvariableop_resource:;
,block5_conv3_biasadd_readvariableop_resource:	G
+block5_conv4_conv2d_readvariableop_resource:;
,block5_conv4_biasadd_readvariableop_resource:	
identity¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp¢#block4_conv2/BiasAdd/ReadVariableOp¢"block4_conv2/Conv2D/ReadVariableOp¢#block4_conv3/BiasAdd/ReadVariableOp¢"block4_conv3/Conv2D/ReadVariableOp¢#block4_conv4/BiasAdd/ReadVariableOp¢"block4_conv4/Conv2D/ReadVariableOp¢#block5_conv1/BiasAdd/ReadVariableOp¢"block5_conv1/Conv2D/ReadVariableOp¢#block5_conv2/BiasAdd/ReadVariableOp¢"block5_conv2/Conv2D/ReadVariableOp¢#block5_conv3/BiasAdd/ReadVariableOp¢"block5_conv3/Conv2D/ReadVariableOp¢#block5_conv4/BiasAdd/ReadVariableOp¢"block5_conv4/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¬
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdds
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdds
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22­
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö	
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
¿

D__inference_model_1_layer_call_and_return_conditional_losses_1038107

inputsK
1vgg19_block1_conv1_conv2d_readvariableop_resource:@@
2vgg19_block1_conv1_biasadd_readvariableop_resource:@K
1vgg19_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg19_block1_conv2_biasadd_readvariableop_resource:@L
1vgg19_block2_conv1_conv2d_readvariableop_resource:@A
2vgg19_block2_conv1_biasadd_readvariableop_resource:	M
1vgg19_block2_conv2_conv2d_readvariableop_resource:A
2vgg19_block2_conv2_biasadd_readvariableop_resource:	M
1vgg19_block3_conv1_conv2d_readvariableop_resource:A
2vgg19_block3_conv1_biasadd_readvariableop_resource:	M
1vgg19_block3_conv2_conv2d_readvariableop_resource:A
2vgg19_block3_conv2_biasadd_readvariableop_resource:	M
1vgg19_block3_conv3_conv2d_readvariableop_resource:A
2vgg19_block3_conv3_biasadd_readvariableop_resource:	M
1vgg19_block3_conv4_conv2d_readvariableop_resource:A
2vgg19_block3_conv4_biasadd_readvariableop_resource:	M
1vgg19_block4_conv1_conv2d_readvariableop_resource:A
2vgg19_block4_conv1_biasadd_readvariableop_resource:	M
1vgg19_block4_conv2_conv2d_readvariableop_resource:A
2vgg19_block4_conv2_biasadd_readvariableop_resource:	M
1vgg19_block4_conv3_conv2d_readvariableop_resource:A
2vgg19_block4_conv3_biasadd_readvariableop_resource:	M
1vgg19_block4_conv4_conv2d_readvariableop_resource:A
2vgg19_block4_conv4_biasadd_readvariableop_resource:	M
1vgg19_block5_conv1_conv2d_readvariableop_resource:A
2vgg19_block5_conv1_biasadd_readvariableop_resource:	M
1vgg19_block5_conv2_conv2d_readvariableop_resource:A
2vgg19_block5_conv2_biasadd_readvariableop_resource:	M
1vgg19_block5_conv3_conv2d_readvariableop_resource:A
2vgg19_block5_conv3_biasadd_readvariableop_resource:	M
1vgg19_block5_conv4_conv2d_readvariableop_resource:A
2vgg19_block5_conv4_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	@5
'dense_1_biasadd_readvariableop_resource:@
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢)vgg19/block1_conv1/BiasAdd/ReadVariableOp¢(vgg19/block1_conv1/Conv2D/ReadVariableOp¢)vgg19/block1_conv2/BiasAdd/ReadVariableOp¢(vgg19/block1_conv2/Conv2D/ReadVariableOp¢)vgg19/block2_conv1/BiasAdd/ReadVariableOp¢(vgg19/block2_conv1/Conv2D/ReadVariableOp¢)vgg19/block2_conv2/BiasAdd/ReadVariableOp¢(vgg19/block2_conv2/Conv2D/ReadVariableOp¢)vgg19/block3_conv1/BiasAdd/ReadVariableOp¢(vgg19/block3_conv1/Conv2D/ReadVariableOp¢)vgg19/block3_conv2/BiasAdd/ReadVariableOp¢(vgg19/block3_conv2/Conv2D/ReadVariableOp¢)vgg19/block3_conv3/BiasAdd/ReadVariableOp¢(vgg19/block3_conv3/Conv2D/ReadVariableOp¢)vgg19/block3_conv4/BiasAdd/ReadVariableOp¢(vgg19/block3_conv4/Conv2D/ReadVariableOp¢)vgg19/block4_conv1/BiasAdd/ReadVariableOp¢(vgg19/block4_conv1/Conv2D/ReadVariableOp¢)vgg19/block4_conv2/BiasAdd/ReadVariableOp¢(vgg19/block4_conv2/Conv2D/ReadVariableOp¢)vgg19/block4_conv3/BiasAdd/ReadVariableOp¢(vgg19/block4_conv3/Conv2D/ReadVariableOp¢)vgg19/block4_conv4/BiasAdd/ReadVariableOp¢(vgg19/block4_conv4/Conv2D/ReadVariableOp¢)vgg19/block5_conv1/BiasAdd/ReadVariableOp¢(vgg19/block5_conv1/Conv2D/ReadVariableOp¢)vgg19/block5_conv2/BiasAdd/ReadVariableOp¢(vgg19/block5_conv2/Conv2D/ReadVariableOp¢)vgg19/block5_conv3/BiasAdd/ReadVariableOp¢(vgg19/block5_conv3/Conv2D/ReadVariableOp¢)vgg19/block5_conv4/BiasAdd/ReadVariableOp¢(vgg19/block5_conv4/Conv2D/ReadVariableOp¢
(vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Á
vgg19/block1_conv1/Conv2DConv2Dinputs0vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

)vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg19/block1_conv1/BiasAddBiasAdd"vgg19/block1_conv1/Conv2D:output:01vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
vgg19/block1_conv1/ReluRelu#vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¢
(vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0à
vgg19/block1_conv2/Conv2DConv2D%vgg19/block1_conv1/Relu:activations:00vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

)vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¸
vgg19/block1_conv2/BiasAddBiasAdd"vgg19/block1_conv2/Conv2D:output:01vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
vgg19/block1_conv2/ReluRelu#vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¸
vgg19/block1_pool/MaxPoolMaxPool%vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
£
(vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
vgg19/block2_conv1/Conv2DConv2D"vgg19/block1_pool/MaxPool:output:00vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

)vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block2_conv1/BiasAddBiasAdd"vgg19/block2_conv1/Conv2D:output:01vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
vgg19/block2_conv1/ReluRelu#vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¤
(vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block2_conv2/Conv2DConv2D%vgg19/block2_conv1/Relu:activations:00vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

)vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block2_conv2/BiasAddBiasAdd"vgg19/block2_conv2/Conv2D:output:01vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
vgg19/block2_conv2/ReluRelu#vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¹
vgg19/block2_pool/MaxPoolMaxPool%vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
¤
(vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block3_conv1/Conv2DConv2D"vgg19/block2_pool/MaxPool:output:00vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv1/BiasAddBiasAdd"vgg19/block3_conv1/Conv2D:output:01vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv1/ReluRelu#vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv2/Conv2DConv2D%vgg19/block3_conv1/Relu:activations:00vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv2/BiasAddBiasAdd"vgg19/block3_conv2/Conv2D:output:01vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv2/ReluRelu#vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv3/Conv2DConv2D%vgg19/block3_conv2/Relu:activations:00vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv3/BiasAddBiasAdd"vgg19/block3_conv3/Conv2D:output:01vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv3/ReluRelu#vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
(vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block3_conv4/Conv2DConv2D%vgg19/block3_conv3/Relu:activations:00vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block3_conv4/BiasAddBiasAdd"vgg19/block3_conv4/Conv2D:output:01vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
vgg19/block3_conv4/ReluRelu#vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¹
vgg19/block3_pool/MaxPoolMaxPool%vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block4_conv1/Conv2DConv2D"vgg19/block3_pool/MaxPool:output:00vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv1/BiasAddBiasAdd"vgg19/block4_conv1/Conv2D:output:01vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv1/ReluRelu#vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv2/Conv2DConv2D%vgg19/block4_conv1/Relu:activations:00vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv2/BiasAddBiasAdd"vgg19/block4_conv2/Conv2D:output:01vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv2/ReluRelu#vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv3/Conv2DConv2D%vgg19/block4_conv2/Relu:activations:00vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv3/BiasAddBiasAdd"vgg19/block4_conv3/Conv2D:output:01vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv3/ReluRelu#vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block4_conv4/Conv2DConv2D%vgg19/block4_conv3/Relu:activations:00vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block4_conv4/BiasAddBiasAdd"vgg19/block4_conv4/Conv2D:output:01vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block4_conv4/ReluRelu#vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg19/block4_pool/MaxPoolMaxPool%vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¤
(vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ü
vgg19/block5_conv1/Conv2DConv2D"vgg19/block4_pool/MaxPool:output:00vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv1/BiasAddBiasAdd"vgg19/block5_conv1/Conv2D:output:01vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv1/ReluRelu#vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv2/Conv2DConv2D%vgg19/block5_conv1/Relu:activations:00vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv2/BiasAddBiasAdd"vgg19/block5_conv2/Conv2D:output:01vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv2/ReluRelu#vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv3/Conv2DConv2D%vgg19/block5_conv2/Relu:activations:00vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv3/BiasAddBiasAdd"vgg19/block5_conv3/Conv2D:output:01vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv3/ReluRelu#vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
(vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp1vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ß
vgg19/block5_conv4/Conv2DConv2D%vgg19/block5_conv3/Relu:activations:00vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

)vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp2vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0·
vgg19/block5_conv4/BiasAddBiasAdd"vgg19/block5_conv4/Conv2D:output:01vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
vgg19/block5_conv4/ReluRelu#vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
vgg19/block5_pool/MaxPoolMaxPool%vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      º
global_average_pooling2d_1/MeanMean"vgg19/block5_pool/MaxPool:output:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_1/MatMulMatMul(global_average_pooling2d_1/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@÷
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^vgg19/block1_conv1/BiasAdd/ReadVariableOp)^vgg19/block1_conv1/Conv2D/ReadVariableOp*^vgg19/block1_conv2/BiasAdd/ReadVariableOp)^vgg19/block1_conv2/Conv2D/ReadVariableOp*^vgg19/block2_conv1/BiasAdd/ReadVariableOp)^vgg19/block2_conv1/Conv2D/ReadVariableOp*^vgg19/block2_conv2/BiasAdd/ReadVariableOp)^vgg19/block2_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv1/BiasAdd/ReadVariableOp)^vgg19/block3_conv1/Conv2D/ReadVariableOp*^vgg19/block3_conv2/BiasAdd/ReadVariableOp)^vgg19/block3_conv2/Conv2D/ReadVariableOp*^vgg19/block3_conv3/BiasAdd/ReadVariableOp)^vgg19/block3_conv3/Conv2D/ReadVariableOp*^vgg19/block3_conv4/BiasAdd/ReadVariableOp)^vgg19/block3_conv4/Conv2D/ReadVariableOp*^vgg19/block4_conv1/BiasAdd/ReadVariableOp)^vgg19/block4_conv1/Conv2D/ReadVariableOp*^vgg19/block4_conv2/BiasAdd/ReadVariableOp)^vgg19/block4_conv2/Conv2D/ReadVariableOp*^vgg19/block4_conv3/BiasAdd/ReadVariableOp)^vgg19/block4_conv3/Conv2D/ReadVariableOp*^vgg19/block4_conv4/BiasAdd/ReadVariableOp)^vgg19/block4_conv4/Conv2D/ReadVariableOp*^vgg19/block5_conv1/BiasAdd/ReadVariableOp)^vgg19/block5_conv1/Conv2D/ReadVariableOp*^vgg19/block5_conv2/BiasAdd/ReadVariableOp)^vgg19/block5_conv2/Conv2D/ReadVariableOp*^vgg19/block5_conv3/BiasAdd/ReadVariableOp)^vgg19/block5_conv3/Conv2D/ReadVariableOp*^vgg19/block5_conv4/BiasAdd/ReadVariableOp)^vgg19/block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)vgg19/block1_conv1/BiasAdd/ReadVariableOp)vgg19/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv1/Conv2D/ReadVariableOp(vgg19/block1_conv1/Conv2D/ReadVariableOp2V
)vgg19/block1_conv2/BiasAdd/ReadVariableOp)vgg19/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block1_conv2/Conv2D/ReadVariableOp(vgg19/block1_conv2/Conv2D/ReadVariableOp2V
)vgg19/block2_conv1/BiasAdd/ReadVariableOp)vgg19/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv1/Conv2D/ReadVariableOp(vgg19/block2_conv1/Conv2D/ReadVariableOp2V
)vgg19/block2_conv2/BiasAdd/ReadVariableOp)vgg19/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block2_conv2/Conv2D/ReadVariableOp(vgg19/block2_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv1/BiasAdd/ReadVariableOp)vgg19/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv1/Conv2D/ReadVariableOp(vgg19/block3_conv1/Conv2D/ReadVariableOp2V
)vgg19/block3_conv2/BiasAdd/ReadVariableOp)vgg19/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv2/Conv2D/ReadVariableOp(vgg19/block3_conv2/Conv2D/ReadVariableOp2V
)vgg19/block3_conv3/BiasAdd/ReadVariableOp)vgg19/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv3/Conv2D/ReadVariableOp(vgg19/block3_conv3/Conv2D/ReadVariableOp2V
)vgg19/block3_conv4/BiasAdd/ReadVariableOp)vgg19/block3_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block3_conv4/Conv2D/ReadVariableOp(vgg19/block3_conv4/Conv2D/ReadVariableOp2V
)vgg19/block4_conv1/BiasAdd/ReadVariableOp)vgg19/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv1/Conv2D/ReadVariableOp(vgg19/block4_conv1/Conv2D/ReadVariableOp2V
)vgg19/block4_conv2/BiasAdd/ReadVariableOp)vgg19/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv2/Conv2D/ReadVariableOp(vgg19/block4_conv2/Conv2D/ReadVariableOp2V
)vgg19/block4_conv3/BiasAdd/ReadVariableOp)vgg19/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv3/Conv2D/ReadVariableOp(vgg19/block4_conv3/Conv2D/ReadVariableOp2V
)vgg19/block4_conv4/BiasAdd/ReadVariableOp)vgg19/block4_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block4_conv4/Conv2D/ReadVariableOp(vgg19/block4_conv4/Conv2D/ReadVariableOp2V
)vgg19/block5_conv1/BiasAdd/ReadVariableOp)vgg19/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv1/Conv2D/ReadVariableOp(vgg19/block5_conv1/Conv2D/ReadVariableOp2V
)vgg19/block5_conv2/BiasAdd/ReadVariableOp)vgg19/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv2/Conv2D/ReadVariableOp(vgg19/block5_conv2/Conv2D/ReadVariableOp2V
)vgg19/block5_conv3/BiasAdd/ReadVariableOp)vgg19/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv3/Conv2D/ReadVariableOp(vgg19/block5_conv3/Conv2D/ReadVariableOp2V
)vgg19/block5_conv4/BiasAdd/ReadVariableOp)vgg19/block5_conv4/BiasAdd/ReadVariableOp2T
(vgg19/block5_conv4/Conv2D/ReadVariableOp(vgg19/block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
þ
¦
.__inference_block3_conv4_layer_call_fn_1038746

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv4_layer_call_and_return_conditional_losses_1034674x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs


I__inference_block4_conv3_layer_call_and_return_conditional_losses_1038827

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block5_conv2_layer_call_and_return_conditional_losses_1038897

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
I
-__inference_block4_pool_layer_call_fn_1038852

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_pool_layer_call_and_return_conditional_losses_1034520
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
£
.__inference_block1_conv1_layer_call_fn_1038586

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block1_conv1_layer_call_and_return_conditional_losses_1034553y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
û
¥
.__inference_block2_conv1_layer_call_fn_1038636

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block2_conv1_layer_call_and_return_conditional_losses_1034588x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@
 
_user_specified_nameinputs


o
C__inference_lambda_layer_call_and_return_conditional_losses_1038133
inputs_0
inputs_1
identityP
subSubinputs_0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
SquareSquaresub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1


I__inference_block5_conv4_layer_call_and_return_conditional_losses_1034812

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
)
§
D__inference_model_2_layer_call_and_return_conditional_losses_1036872
input_3
input_4)
model_1_1036761:@
model_1_1036763:@)
model_1_1036765:@@
model_1_1036767:@*
model_1_1036769:@
model_1_1036771:	+
model_1_1036773:
model_1_1036775:	+
model_1_1036777:
model_1_1036779:	+
model_1_1036781:
model_1_1036783:	+
model_1_1036785:
model_1_1036787:	+
model_1_1036789:
model_1_1036791:	+
model_1_1036793:
model_1_1036795:	+
model_1_1036797:
model_1_1036799:	+
model_1_1036801:
model_1_1036803:	+
model_1_1036805:
model_1_1036807:	+
model_1_1036809:
model_1_1036811:	+
model_1_1036813:
model_1_1036815:	+
model_1_1036817:
model_1_1036819:	+
model_1_1036821:
model_1_1036823:	"
model_1_1036825:	@
model_1_1036827:@!
dense_2_1036866:
dense_2_1036868:
identity¢dense_2/StatefulPartitionedCall¢model_1/StatefulPartitionedCall¢!model_1/StatefulPartitionedCall_1Ó
model_1/StatefulPartitionedCallStatefulPartitionedCallinput_3model_1_1036761model_1_1036763model_1_1036765model_1_1036767model_1_1036769model_1_1036771model_1_1036773model_1_1036775model_1_1036777model_1_1036779model_1_1036781model_1_1036783model_1_1036785model_1_1036787model_1_1036789model_1_1036791model_1_1036793model_1_1036795model_1_1036797model_1_1036799model_1_1036801model_1_1036803model_1_1036805model_1_1036807model_1_1036809model_1_1036811model_1_1036813model_1_1036815model_1_1036817model_1_1036819model_1_1036821model_1_1036823model_1_1036825model_1_1036827*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035624Õ
!model_1/StatefulPartitionedCall_1StatefulPartitionedCallinput_4model_1_1036761model_1_1036763model_1_1036765model_1_1036767model_1_1036769model_1_1036771model_1_1036773model_1_1036775model_1_1036777model_1_1036779model_1_1036781model_1_1036783model_1_1036785model_1_1036787model_1_1036789model_1_1036791model_1_1036793model_1_1036795model_1_1036797model_1_1036799model_1_1036801model_1_1036803model_1_1036805model_1_1036807model_1_1036809model_1_1036811model_1_1036813model_1_1036815model_1_1036817model_1_1036819model_1_1036821model_1_1036823model_1_1036825model_1_1036827*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035624
lambda/PartitionedCallPartitionedCall(model_1/StatefulPartitionedCall:output:0*model_1/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_1036276
dense_2/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_2_1036866dense_2_1036868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1036289w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp ^dense_2/StatefulPartitionedCall ^model_1/StatefulPartitionedCall"^model_1/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall2F
!model_1/StatefulPartitionedCall_1!model_1/StatefulPartitionedCall_1:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_3:ZV
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_4


I__inference_block3_conv4_layer_call_and_return_conditional_losses_1038757

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Å

)__inference_dense_2_layer_call_fn_1038156

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1036289o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block3_conv2_layer_call_and_return_conditional_losses_1034640

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs

d
H__inference_block5_pool_layer_call_and_return_conditional_losses_1034532

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block3_conv3_layer_call_and_return_conditional_losses_1034657

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
µ
I
-__inference_block1_pool_layer_call_fn_1038622

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_pool_layer_call_and_return_conditional_losses_1034484
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block3_conv2_layer_call_and_return_conditional_losses_1038717

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
µ
I
-__inference_block5_pool_layer_call_fn_1038942

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_pool_layer_call_and_return_conditional_losses_1034532
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
¦
.__inference_block5_conv1_layer_call_fn_1038866

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv1_layer_call_and_return_conditional_losses_1034761x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
)
§
D__inference_model_2_layer_call_and_return_conditional_losses_1036296

inputs
inputs_1)
model_1_1036158:@
model_1_1036160:@)
model_1_1036162:@@
model_1_1036164:@*
model_1_1036166:@
model_1_1036168:	+
model_1_1036170:
model_1_1036172:	+
model_1_1036174:
model_1_1036176:	+
model_1_1036178:
model_1_1036180:	+
model_1_1036182:
model_1_1036184:	+
model_1_1036186:
model_1_1036188:	+
model_1_1036190:
model_1_1036192:	+
model_1_1036194:
model_1_1036196:	+
model_1_1036198:
model_1_1036200:	+
model_1_1036202:
model_1_1036204:	+
model_1_1036206:
model_1_1036208:	+
model_1_1036210:
model_1_1036212:	+
model_1_1036214:
model_1_1036216:	+
model_1_1036218:
model_1_1036220:	"
model_1_1036222:	@
model_1_1036224:@!
dense_2_1036290:
dense_2_1036292:
identity¢dense_2/StatefulPartitionedCall¢model_1/StatefulPartitionedCall¢!model_1/StatefulPartitionedCall_1Ò
model_1/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_1_1036158model_1_1036160model_1_1036162model_1_1036164model_1_1036166model_1_1036168model_1_1036170model_1_1036172model_1_1036174model_1_1036176model_1_1036178model_1_1036180model_1_1036182model_1_1036184model_1_1036186model_1_1036188model_1_1036190model_1_1036192model_1_1036194model_1_1036196model_1_1036198model_1_1036200model_1_1036202model_1_1036204model_1_1036206model_1_1036208model_1_1036210model_1_1036212model_1_1036214model_1_1036216model_1_1036218model_1_1036220model_1_1036222model_1_1036224*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035624Ö
!model_1/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_1_1036158model_1_1036160model_1_1036162model_1_1036164model_1_1036166model_1_1036168model_1_1036170model_1_1036172model_1_1036174model_1_1036176model_1_1036178model_1_1036180model_1_1036182model_1_1036184model_1_1036186model_1_1036188model_1_1036190model_1_1036192model_1_1036194model_1_1036196model_1_1036198model_1_1036200model_1_1036202model_1_1036204model_1_1036206model_1_1036208model_1_1036210model_1_1036212model_1_1036214model_1_1036216model_1_1036218model_1_1036220model_1_1036222model_1_1036224*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035624
lambda/PartitionedCallPartitionedCall(model_1/StatefulPartitionedCall:output:0*model_1/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_1036276
dense_2/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_2_1036290dense_2_1036292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1036289w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp ^dense_2/StatefulPartitionedCall ^model_1/StatefulPartitionedCall"^model_1/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall2F
!model_1/StatefulPartitionedCall_1!model_1/StatefulPartitionedCall_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
þ
¦
.__inference_block5_conv2_layer_call_fn_1038886

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv2_layer_call_and_return_conditional_losses_1034778x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
ý	
)__inference_model_2_layer_call_fn_1036371
input_3
input_4!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:

unknown_34:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_1036296o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_3:ZV
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_4
Ë	
ö
D__inference_dense_1_layer_call_and_return_conditional_losses_1035617

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
¦
.__inference_block3_conv3_layer_call_fn_1038726

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv3_layer_call_and_return_conditional_losses_1034657x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ùg
º
B__inference_vgg19_layer_call_and_return_conditional_losses_1034820

inputs.
block1_conv1_1034554:@"
block1_conv1_1034556:@.
block1_conv2_1034571:@@"
block1_conv2_1034573:@/
block2_conv1_1034589:@#
block2_conv1_1034591:	0
block2_conv2_1034606:#
block2_conv2_1034608:	0
block3_conv1_1034624:#
block3_conv1_1034626:	0
block3_conv2_1034641:#
block3_conv2_1034643:	0
block3_conv3_1034658:#
block3_conv3_1034660:	0
block3_conv4_1034675:#
block3_conv4_1034677:	0
block4_conv1_1034693:#
block4_conv1_1034695:	0
block4_conv2_1034710:#
block4_conv2_1034712:	0
block4_conv3_1034727:#
block4_conv3_1034729:	0
block4_conv4_1034744:#
block4_conv4_1034746:	0
block5_conv1_1034762:#
block5_conv1_1034764:	0
block5_conv2_1034779:#
block5_conv2_1034781:	0
block5_conv3_1034796:#
block5_conv3_1034798:	0
block5_conv4_1034813:#
block5_conv4_1034815:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_1034554block1_conv1_1034556*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block1_conv1_layer_call_and_return_conditional_losses_1034553·
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_1034571block1_conv2_1034573*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block1_conv2_layer_call_and_return_conditional_losses_1034570ñ
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_pool_layer_call_and_return_conditional_losses_1034484­
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_1034589block2_conv1_1034591*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block2_conv1_layer_call_and_return_conditional_losses_1034588¶
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_1034606block2_conv2_1034608*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block2_conv2_layer_call_and_return_conditional_losses_1034605ò
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_pool_layer_call_and_return_conditional_losses_1034496­
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_1034624block3_conv1_1034626*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv1_layer_call_and_return_conditional_losses_1034623¶
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_1034641block3_conv2_1034643*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv2_layer_call_and_return_conditional_losses_1034640¶
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_1034658block3_conv3_1034660*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv3_layer_call_and_return_conditional_losses_1034657¶
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_1034675block3_conv4_1034677*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv4_layer_call_and_return_conditional_losses_1034674ò
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_pool_layer_call_and_return_conditional_losses_1034508­
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_1034693block4_conv1_1034695*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv1_layer_call_and_return_conditional_losses_1034692¶
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_1034710block4_conv2_1034712*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv2_layer_call_and_return_conditional_losses_1034709¶
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_1034727block4_conv3_1034729*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv3_layer_call_and_return_conditional_losses_1034726¶
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_1034744block4_conv4_1034746*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv4_layer_call_and_return_conditional_losses_1034743ò
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_pool_layer_call_and_return_conditional_losses_1034520­
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_1034762block5_conv1_1034764*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv1_layer_call_and_return_conditional_losses_1034761¶
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_1034779block5_conv2_1034781*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv2_layer_call_and_return_conditional_losses_1034778¶
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_1034796block5_conv3_1034798*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv3_layer_call_and_return_conditional_losses_1034795¶
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_1034813block5_conv4_1034815*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv4_layer_call_and_return_conditional_losses_1034812ò
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_pool_layer_call_and_return_conditional_losses_1034532|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs


I__inference_block1_conv1_layer_call_and_return_conditional_losses_1034553

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
þ
¦
.__inference_block4_conv4_layer_call_fn_1038836

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv4_layer_call_and_return_conditional_losses_1034743x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
)
§
D__inference_model_2_layer_call_and_return_conditional_losses_1036987
input_3
input_4)
model_1_1036876:@
model_1_1036878:@)
model_1_1036880:@@
model_1_1036882:@*
model_1_1036884:@
model_1_1036886:	+
model_1_1036888:
model_1_1036890:	+
model_1_1036892:
model_1_1036894:	+
model_1_1036896:
model_1_1036898:	+
model_1_1036900:
model_1_1036902:	+
model_1_1036904:
model_1_1036906:	+
model_1_1036908:
model_1_1036910:	+
model_1_1036912:
model_1_1036914:	+
model_1_1036916:
model_1_1036918:	+
model_1_1036920:
model_1_1036922:	+
model_1_1036924:
model_1_1036926:	+
model_1_1036928:
model_1_1036930:	+
model_1_1036932:
model_1_1036934:	+
model_1_1036936:
model_1_1036938:	"
model_1_1036940:	@
model_1_1036942:@!
dense_2_1036981:
dense_2_1036983:
identity¢dense_2/StatefulPartitionedCall¢model_1/StatefulPartitionedCall¢!model_1/StatefulPartitionedCall_1Ó
model_1/StatefulPartitionedCallStatefulPartitionedCallinput_3model_1_1036876model_1_1036878model_1_1036880model_1_1036882model_1_1036884model_1_1036886model_1_1036888model_1_1036890model_1_1036892model_1_1036894model_1_1036896model_1_1036898model_1_1036900model_1_1036902model_1_1036904model_1_1036906model_1_1036908model_1_1036910model_1_1036912model_1_1036914model_1_1036916model_1_1036918model_1_1036920model_1_1036922model_1_1036924model_1_1036926model_1_1036928model_1_1036930model_1_1036932model_1_1036934model_1_1036936model_1_1036938model_1_1036940model_1_1036942*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035855Õ
!model_1/StatefulPartitionedCall_1StatefulPartitionedCallinput_4model_1_1036876model_1_1036878model_1_1036880model_1_1036882model_1_1036884model_1_1036886model_1_1036888model_1_1036890model_1_1036892model_1_1036894model_1_1036896model_1_1036898model_1_1036900model_1_1036902model_1_1036904model_1_1036906model_1_1036908model_1_1036910model_1_1036912model_1_1036914model_1_1036916model_1_1036918model_1_1036920model_1_1036922model_1_1036924model_1_1036926model_1_1036928model_1_1036930model_1_1036932model_1_1036934model_1_1036936model_1_1036938model_1_1036940model_1_1036942*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035855
lambda/PartitionedCallPartitionedCall(model_1/StatefulPartitionedCall:output:0*model_1/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_1036405
dense_2/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_2_1036981dense_2_1036983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1036289w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp ^dense_2/StatefulPartitionedCall ^model_1/StatefulPartitionedCall"^model_1/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall2F
!model_1/StatefulPartitionedCall_1!model_1/StatefulPartitionedCall_1:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_3:ZV
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_4
°
ñ

D__inference_model_1_layer_call_and_return_conditional_losses_1035624

inputs'
vgg19_1035541:@
vgg19_1035543:@'
vgg19_1035545:@@
vgg19_1035547:@(
vgg19_1035549:@
vgg19_1035551:	)
vgg19_1035553:
vgg19_1035555:	)
vgg19_1035557:
vgg19_1035559:	)
vgg19_1035561:
vgg19_1035563:	)
vgg19_1035565:
vgg19_1035567:	)
vgg19_1035569:
vgg19_1035571:	)
vgg19_1035573:
vgg19_1035575:	)
vgg19_1035577:
vgg19_1035579:	)
vgg19_1035581:
vgg19_1035583:	)
vgg19_1035585:
vgg19_1035587:	)
vgg19_1035589:
vgg19_1035591:	)
vgg19_1035593:
vgg19_1035595:	)
vgg19_1035597:
vgg19_1035599:	)
vgg19_1035601:
vgg19_1035603:	"
dense_1_1035618:	@
dense_1_1035620:@
identity¢dense_1/StatefulPartitionedCall¢vgg19/StatefulPartitionedCallñ
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputsvgg19_1035541vgg19_1035543vgg19_1035545vgg19_1035547vgg19_1035549vgg19_1035551vgg19_1035553vgg19_1035555vgg19_1035557vgg19_1035559vgg19_1035561vgg19_1035563vgg19_1035565vgg19_1035567vgg19_1035569vgg19_1035571vgg19_1035573vgg19_1035575vgg19_1035577vgg19_1035579vgg19_1035581vgg19_1035583vgg19_1035585vgg19_1035587vgg19_1035589vgg19_1035591vgg19_1035593vgg19_1035595vgg19_1035597vgg19_1035599vgg19_1035601vgg19_1035603*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_vgg19_layer_call_and_return_conditional_losses_1034820
*global_average_pooling2d_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_1035531
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0dense_1_1035618dense_1_1035620*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1035617w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp ^dense_1/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
¹
ù	
%__inference_signature_wrapper_1037703
input_3
input_4!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_1034475o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_3:ZV
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_4
¤Ê
û 
#__inference__traced_restore_1039299
file_prefix1
assignvariableop_dense_2_kernel:-
assignvariableop_1_dense_2_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: @
&assignvariableop_7_block1_conv1_kernel:@2
$assignvariableop_8_block1_conv1_bias:@@
&assignvariableop_9_block1_conv2_kernel:@@3
%assignvariableop_10_block1_conv2_bias:@B
'assignvariableop_11_block2_conv1_kernel:@4
%assignvariableop_12_block2_conv1_bias:	C
'assignvariableop_13_block2_conv2_kernel:4
%assignvariableop_14_block2_conv2_bias:	C
'assignvariableop_15_block3_conv1_kernel:4
%assignvariableop_16_block3_conv1_bias:	C
'assignvariableop_17_block3_conv2_kernel:4
%assignvariableop_18_block3_conv2_bias:	C
'assignvariableop_19_block3_conv3_kernel:4
%assignvariableop_20_block3_conv3_bias:	C
'assignvariableop_21_block3_conv4_kernel:4
%assignvariableop_22_block3_conv4_bias:	C
'assignvariableop_23_block4_conv1_kernel:4
%assignvariableop_24_block4_conv1_bias:	C
'assignvariableop_25_block4_conv2_kernel:4
%assignvariableop_26_block4_conv2_bias:	C
'assignvariableop_27_block4_conv3_kernel:4
%assignvariableop_28_block4_conv3_bias:	C
'assignvariableop_29_block4_conv4_kernel:4
%assignvariableop_30_block4_conv4_bias:	C
'assignvariableop_31_block5_conv1_kernel:4
%assignvariableop_32_block5_conv1_bias:	C
'assignvariableop_33_block5_conv2_kernel:4
%assignvariableop_34_block5_conv2_bias:	C
'assignvariableop_35_block5_conv3_kernel:4
%assignvariableop_36_block5_conv3_bias:	C
'assignvariableop_37_block5_conv4_kernel:4
%assignvariableop_38_block5_conv4_bias:	5
"assignvariableop_39_dense_1_kernel:	@.
 assignvariableop_40_dense_1_bias:@#
assignvariableop_41_total: #
assignvariableop_42_count: %
assignvariableop_43_total_1: %
assignvariableop_44_count_1: ;
)assignvariableop_45_adam_dense_2_kernel_m:5
'assignvariableop_46_adam_dense_2_bias_m:<
)assignvariableop_47_adam_dense_1_kernel_m:	@5
'assignvariableop_48_adam_dense_1_bias_m:@;
)assignvariableop_49_adam_dense_2_kernel_v:5
'assignvariableop_50_adam_dense_2_bias_v:<
)assignvariableop_51_adam_dense_1_kernel_v:	@5
'assignvariableop_52_adam_dense_1_bias_v:@
identity_54¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*¦
valueB6B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÜ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¯
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*î
_output_shapesÛ
Ø::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp&assignvariableop_7_block1_conv1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp$assignvariableop_8_block1_conv1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp&assignvariableop_9_block1_conv2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp%assignvariableop_10_block1_conv2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp'assignvariableop_11_block2_conv1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp%assignvariableop_12_block2_conv1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp'assignvariableop_13_block2_conv2_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp%assignvariableop_14_block2_conv2_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp'assignvariableop_15_block3_conv1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp%assignvariableop_16_block3_conv1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp'assignvariableop_17_block3_conv2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp%assignvariableop_18_block3_conv2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp'assignvariableop_19_block3_conv3_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp%assignvariableop_20_block3_conv3_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp'assignvariableop_21_block3_conv4_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp%assignvariableop_22_block3_conv4_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp'assignvariableop_23_block4_conv1_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp%assignvariableop_24_block4_conv1_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp'assignvariableop_25_block4_conv2_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp%assignvariableop_26_block4_conv2_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp'assignvariableop_27_block4_conv3_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp%assignvariableop_28_block4_conv3_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp'assignvariableop_29_block4_conv4_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp%assignvariableop_30_block4_conv4_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp'assignvariableop_31_block5_conv1_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp%assignvariableop_32_block5_conv1_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp'assignvariableop_33_block5_conv2_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp%assignvariableop_34_block5_conv2_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp'assignvariableop_35_block5_conv3_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp%assignvariableop_36_block5_conv3_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp'assignvariableop_37_block5_conv4_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp%assignvariableop_38_block5_conv4_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp"assignvariableop_39_dense_1_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp assignvariableop_40_dense_1_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOpassignvariableop_43_total_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_2_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_2_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_2_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_2_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_1_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_1_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ý	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_54IdentityIdentity_53:output:0^NoOp_1*
T0*
_output_shapes
: Ê	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_54Identity_54:output:0*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¸
s
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_1038558

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block4_conv1_layer_call_and_return_conditional_losses_1034692

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
s
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_1035531

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
ý	
)__inference_model_2_layer_call_fn_1036757
input_3
input_4!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:

unknown_34:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_1036604o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_3:ZV
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_4


I__inference_block2_conv2_layer_call_and_return_conditional_losses_1038667

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿdd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs


m
C__inference_lambda_layer_call_and_return_conditional_losses_1036276

inputs
inputs_1
identityN
subSubinputsinputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@K
SquareSquaresub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
©
ý
'__inference_vgg19_layer_call_fn_1034887
input_6!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_vgg19_layer_call_and_return_conditional_losses_1034820x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_6
ñ
·	
)__inference_model_1_layer_call_fn_1037849

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035855o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
þ
¦
.__inference_block3_conv1_layer_call_fn_1038686

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv1_layer_call_and_return_conditional_losses_1034623x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
°
ñ

D__inference_model_1_layer_call_and_return_conditional_losses_1035855

inputs'
vgg19_1035783:@
vgg19_1035785:@'
vgg19_1035787:@@
vgg19_1035789:@(
vgg19_1035791:@
vgg19_1035793:	)
vgg19_1035795:
vgg19_1035797:	)
vgg19_1035799:
vgg19_1035801:	)
vgg19_1035803:
vgg19_1035805:	)
vgg19_1035807:
vgg19_1035809:	)
vgg19_1035811:
vgg19_1035813:	)
vgg19_1035815:
vgg19_1035817:	)
vgg19_1035819:
vgg19_1035821:	)
vgg19_1035823:
vgg19_1035825:	)
vgg19_1035827:
vgg19_1035829:	)
vgg19_1035831:
vgg19_1035833:	)
vgg19_1035835:
vgg19_1035837:	)
vgg19_1035839:
vgg19_1035841:	)
vgg19_1035843:
vgg19_1035845:	"
dense_1_1035849:	@
dense_1_1035851:@
identity¢dense_1/StatefulPartitionedCall¢vgg19/StatefulPartitionedCallñ
vgg19/StatefulPartitionedCallStatefulPartitionedCallinputsvgg19_1035783vgg19_1035785vgg19_1035787vgg19_1035789vgg19_1035791vgg19_1035793vgg19_1035795vgg19_1035797vgg19_1035799vgg19_1035801vgg19_1035803vgg19_1035805vgg19_1035807vgg19_1035809vgg19_1035811vgg19_1035813vgg19_1035815vgg19_1035817vgg19_1035819vgg19_1035821vgg19_1035823vgg19_1035825vgg19_1035827vgg19_1035829vgg19_1035831vgg19_1035833vgg19_1035835vgg19_1035837vgg19_1035839vgg19_1035841vgg19_1035843vgg19_1035845*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_vgg19_layer_call_and_return_conditional_losses_1035207
*global_average_pooling2d_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_1035531
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0dense_1_1035849dense_1_1035851*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1035617w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp ^dense_1/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
þ
¦
.__inference_block4_conv3_layer_call_fn_1038816

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv3_layer_call_and_return_conditional_losses_1034726x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block5_conv1_layer_call_and_return_conditional_losses_1034761

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block2_conv2_layer_call_and_return_conditional_losses_1034605

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿdd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs


I__inference_block4_conv3_layer_call_and_return_conditional_losses_1034726

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block1_conv2_layer_call_and_return_conditional_losses_1038617

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
 
_user_specified_nameinputs


I__inference_block2_conv1_layer_call_and_return_conditional_losses_1038647

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@
 
_user_specified_nameinputs
£
T
(__inference_lambda_layer_call_fn_1038119
inputs_0
inputs_1
identity¾
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_1036405`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ@:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/1
¦
ü
'__inference_vgg19_layer_call_fn_1038236

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_vgg19_layer_call_and_return_conditional_losses_1034820x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs

d
H__inference_block2_pool_layer_call_and_return_conditional_losses_1034496

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block5_conv4_layer_call_and_return_conditional_losses_1038937

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
I
-__inference_block3_pool_layer_call_fn_1038762

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_pool_layer_call_and_return_conditional_losses_1034508
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
¸	
)__inference_model_1_layer_call_fn_1035999
input_5!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035855o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_5


I__inference_block4_conv2_layer_call_and_return_conditional_losses_1038807

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


õ
D__inference_dense_2_layer_call_and_return_conditional_losses_1036289

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë	
ö
D__inference_dense_1_layer_call_and_return_conditional_losses_1038577

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
¦
.__inference_block4_conv1_layer_call_fn_1038776

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv1_layer_call_and_return_conditional_losses_1034692x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
·	
)__inference_model_1_layer_call_fn_1037776

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035624o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
È

)__inference_dense_1_layer_call_fn_1038567

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1035617o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
H__inference_block1_pool_layer_call_and_return_conditional_losses_1038627

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block5_conv3_layer_call_and_return_conditional_losses_1038917

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
ò

D__inference_model_1_layer_call_and_return_conditional_losses_1036074
input_5'
vgg19_1036002:@
vgg19_1036004:@'
vgg19_1036006:@@
vgg19_1036008:@(
vgg19_1036010:@
vgg19_1036012:	)
vgg19_1036014:
vgg19_1036016:	)
vgg19_1036018:
vgg19_1036020:	)
vgg19_1036022:
vgg19_1036024:	)
vgg19_1036026:
vgg19_1036028:	)
vgg19_1036030:
vgg19_1036032:	)
vgg19_1036034:
vgg19_1036036:	)
vgg19_1036038:
vgg19_1036040:	)
vgg19_1036042:
vgg19_1036044:	)
vgg19_1036046:
vgg19_1036048:	)
vgg19_1036050:
vgg19_1036052:	)
vgg19_1036054:
vgg19_1036056:	)
vgg19_1036058:
vgg19_1036060:	)
vgg19_1036062:
vgg19_1036064:	"
dense_1_1036068:	@
dense_1_1036070:@
identity¢dense_1/StatefulPartitionedCall¢vgg19/StatefulPartitionedCallò
vgg19/StatefulPartitionedCallStatefulPartitionedCallinput_5vgg19_1036002vgg19_1036004vgg19_1036006vgg19_1036008vgg19_1036010vgg19_1036012vgg19_1036014vgg19_1036016vgg19_1036018vgg19_1036020vgg19_1036022vgg19_1036024vgg19_1036026vgg19_1036028vgg19_1036030vgg19_1036032vgg19_1036034vgg19_1036036vgg19_1036038vgg19_1036040vgg19_1036042vgg19_1036044vgg19_1036046vgg19_1036048vgg19_1036050vgg19_1036052vgg19_1036054vgg19_1036056vgg19_1036058vgg19_1036060vgg19_1036062vgg19_1036064*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_vgg19_layer_call_and_return_conditional_losses_1034820
*global_average_pooling2d_1/PartitionedCallPartitionedCall&vgg19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *`
f[RY
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_1035531
dense_1/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0dense_1_1036068dense_1_1036070*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1035617w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp ^dense_1/StatefulPartitionedCall^vgg19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
vgg19/StatefulPartitionedCallvgg19/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_5
üg
»
B__inference_vgg19_layer_call_and_return_conditional_losses_1035521
input_6.
block1_conv1_1035435:@"
block1_conv1_1035437:@.
block1_conv2_1035440:@@"
block1_conv2_1035442:@/
block2_conv1_1035446:@#
block2_conv1_1035448:	0
block2_conv2_1035451:#
block2_conv2_1035453:	0
block3_conv1_1035457:#
block3_conv1_1035459:	0
block3_conv2_1035462:#
block3_conv2_1035464:	0
block3_conv3_1035467:#
block3_conv3_1035469:	0
block3_conv4_1035472:#
block3_conv4_1035474:	0
block4_conv1_1035478:#
block4_conv1_1035480:	0
block4_conv2_1035483:#
block4_conv2_1035485:	0
block4_conv3_1035488:#
block4_conv3_1035490:	0
block4_conv4_1035493:#
block4_conv4_1035495:	0
block5_conv1_1035499:#
block5_conv1_1035501:	0
block5_conv2_1035504:#
block5_conv2_1035506:	0
block5_conv3_1035509:#
block5_conv3_1035511:	0
block5_conv4_1035514:#
block5_conv4_1035516:	
identity¢$block1_conv1/StatefulPartitionedCall¢$block1_conv2/StatefulPartitionedCall¢$block2_conv1/StatefulPartitionedCall¢$block2_conv2/StatefulPartitionedCall¢$block3_conv1/StatefulPartitionedCall¢$block3_conv2/StatefulPartitionedCall¢$block3_conv3/StatefulPartitionedCall¢$block3_conv4/StatefulPartitionedCall¢$block4_conv1/StatefulPartitionedCall¢$block4_conv2/StatefulPartitionedCall¢$block4_conv3/StatefulPartitionedCall¢$block4_conv4/StatefulPartitionedCall¢$block5_conv1/StatefulPartitionedCall¢$block5_conv2/StatefulPartitionedCall¢$block5_conv3/StatefulPartitionedCall¢$block5_conv4/StatefulPartitionedCall
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_6block1_conv1_1035435block1_conv1_1035437*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block1_conv1_layer_call_and_return_conditional_losses_1034553·
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_1035440block1_conv2_1035442*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block1_conv2_layer_call_and_return_conditional_losses_1034570ñ
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block1_pool_layer_call_and_return_conditional_losses_1034484­
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_1035446block2_conv1_1035448*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block2_conv1_layer_call_and_return_conditional_losses_1034588¶
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_1035451block2_conv2_1035453*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block2_conv2_layer_call_and_return_conditional_losses_1034605ò
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_pool_layer_call_and_return_conditional_losses_1034496­
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_1035457block3_conv1_1035459*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv1_layer_call_and_return_conditional_losses_1034623¶
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_1035462block3_conv2_1035464*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv2_layer_call_and_return_conditional_losses_1034640¶
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_1035467block3_conv3_1035469*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv3_layer_call_and_return_conditional_losses_1034657¶
$block3_conv4/StatefulPartitionedCallStatefulPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0block3_conv4_1035472block3_conv4_1035474*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block3_conv4_layer_call_and_return_conditional_losses_1034674ò
block3_pool/PartitionedCallPartitionedCall-block3_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block3_pool_layer_call_and_return_conditional_losses_1034508­
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_1035478block4_conv1_1035480*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv1_layer_call_and_return_conditional_losses_1034692¶
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_1035483block4_conv2_1035485*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv2_layer_call_and_return_conditional_losses_1034709¶
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_1035488block4_conv3_1035490*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv3_layer_call_and_return_conditional_losses_1034726¶
$block4_conv4/StatefulPartitionedCallStatefulPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0block4_conv4_1035493block4_conv4_1035495*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv4_layer_call_and_return_conditional_losses_1034743ò
block4_pool/PartitionedCallPartitionedCall-block4_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block4_pool_layer_call_and_return_conditional_losses_1034520­
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_1035499block5_conv1_1035501*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv1_layer_call_and_return_conditional_losses_1034761¶
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_1035504block5_conv2_1035506*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv2_layer_call_and_return_conditional_losses_1034778¶
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_1035509block5_conv3_1035511*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv3_layer_call_and_return_conditional_losses_1034795¶
$block5_conv4/StatefulPartitionedCallStatefulPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0block5_conv4_1035514block5_conv4_1035516*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv4_layer_call_and_return_conditional_losses_1034812ò
block5_pool/PartitionedCallPartitionedCall-block5_conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block5_pool_layer_call_and_return_conditional_losses_1034532|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block3_conv4/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block4_conv4/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall%^block5_conv4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block3_conv4/StatefulPartitionedCall$block3_conv4/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block4_conv4/StatefulPartitionedCall$block4_conv4/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall2L
$block5_conv4/StatefulPartitionedCall$block5_conv4/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_6

d
H__inference_block3_pool_layer_call_and_return_conditional_losses_1038767

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
)
§
D__inference_model_2_layer_call_and_return_conditional_losses_1036604

inputs
inputs_1)
model_1_1036493:@
model_1_1036495:@)
model_1_1036497:@@
model_1_1036499:@*
model_1_1036501:@
model_1_1036503:	+
model_1_1036505:
model_1_1036507:	+
model_1_1036509:
model_1_1036511:	+
model_1_1036513:
model_1_1036515:	+
model_1_1036517:
model_1_1036519:	+
model_1_1036521:
model_1_1036523:	+
model_1_1036525:
model_1_1036527:	+
model_1_1036529:
model_1_1036531:	+
model_1_1036533:
model_1_1036535:	+
model_1_1036537:
model_1_1036539:	+
model_1_1036541:
model_1_1036543:	+
model_1_1036545:
model_1_1036547:	+
model_1_1036549:
model_1_1036551:	+
model_1_1036553:
model_1_1036555:	"
model_1_1036557:	@
model_1_1036559:@!
dense_2_1036598:
dense_2_1036600:
identity¢dense_2/StatefulPartitionedCall¢model_1/StatefulPartitionedCall¢!model_1/StatefulPartitionedCall_1Ò
model_1/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_1_1036493model_1_1036495model_1_1036497model_1_1036499model_1_1036501model_1_1036503model_1_1036505model_1_1036507model_1_1036509model_1_1036511model_1_1036513model_1_1036515model_1_1036517model_1_1036519model_1_1036521model_1_1036523model_1_1036525model_1_1036527model_1_1036529model_1_1036531model_1_1036533model_1_1036535model_1_1036537model_1_1036539model_1_1036541model_1_1036543model_1_1036545model_1_1036547model_1_1036549model_1_1036551model_1_1036553model_1_1036555model_1_1036557model_1_1036559*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035855Ö
!model_1/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1model_1_1036493model_1_1036495model_1_1036497model_1_1036499model_1_1036501model_1_1036503model_1_1036505model_1_1036507model_1_1036509model_1_1036511model_1_1036513model_1_1036515model_1_1036517model_1_1036519model_1_1036521model_1_1036523model_1_1036525model_1_1036527model_1_1036529model_1_1036531model_1_1036533model_1_1036535model_1_1036537model_1_1036539model_1_1036541model_1_1036543model_1_1036545model_1_1036547model_1_1036549model_1_1036551model_1_1036553model_1_1036555model_1_1036557model_1_1036559*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*D
_read_only_resource_inputs&
$"	
 !"*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_1035855
lambda/PartitionedCallPartitionedCall(model_1/StatefulPartitionedCall:output:0*model_1/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_1036405
dense_2/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0dense_2_1036598dense_2_1036600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1036289w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp ^dense_2/StatefulPartitionedCall ^model_1/StatefulPartitionedCall"^model_1/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall2F
!model_1/StatefulPartitionedCall_1!model_1/StatefulPartitionedCall_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
þ
¦
.__inference_block4_conv2_layer_call_fn_1038796

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block4_conv2_layer_call_and_return_conditional_losses_1034709x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block3_conv1_layer_call_and_return_conditional_losses_1038697

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Àª
3
D__inference_model_2_layer_call_and_return_conditional_losses_1037386
inputs_0
inputs_1S
9model_1_vgg19_block1_conv1_conv2d_readvariableop_resource:@H
:model_1_vgg19_block1_conv1_biasadd_readvariableop_resource:@S
9model_1_vgg19_block1_conv2_conv2d_readvariableop_resource:@@H
:model_1_vgg19_block1_conv2_biasadd_readvariableop_resource:@T
9model_1_vgg19_block2_conv1_conv2d_readvariableop_resource:@I
:model_1_vgg19_block2_conv1_biasadd_readvariableop_resource:	U
9model_1_vgg19_block2_conv2_conv2d_readvariableop_resource:I
:model_1_vgg19_block2_conv2_biasadd_readvariableop_resource:	U
9model_1_vgg19_block3_conv1_conv2d_readvariableop_resource:I
:model_1_vgg19_block3_conv1_biasadd_readvariableop_resource:	U
9model_1_vgg19_block3_conv2_conv2d_readvariableop_resource:I
:model_1_vgg19_block3_conv2_biasadd_readvariableop_resource:	U
9model_1_vgg19_block3_conv3_conv2d_readvariableop_resource:I
:model_1_vgg19_block3_conv3_biasadd_readvariableop_resource:	U
9model_1_vgg19_block3_conv4_conv2d_readvariableop_resource:I
:model_1_vgg19_block3_conv4_biasadd_readvariableop_resource:	U
9model_1_vgg19_block4_conv1_conv2d_readvariableop_resource:I
:model_1_vgg19_block4_conv1_biasadd_readvariableop_resource:	U
9model_1_vgg19_block4_conv2_conv2d_readvariableop_resource:I
:model_1_vgg19_block4_conv2_biasadd_readvariableop_resource:	U
9model_1_vgg19_block4_conv3_conv2d_readvariableop_resource:I
:model_1_vgg19_block4_conv3_biasadd_readvariableop_resource:	U
9model_1_vgg19_block4_conv4_conv2d_readvariableop_resource:I
:model_1_vgg19_block4_conv4_biasadd_readvariableop_resource:	U
9model_1_vgg19_block5_conv1_conv2d_readvariableop_resource:I
:model_1_vgg19_block5_conv1_biasadd_readvariableop_resource:	U
9model_1_vgg19_block5_conv2_conv2d_readvariableop_resource:I
:model_1_vgg19_block5_conv2_biasadd_readvariableop_resource:	U
9model_1_vgg19_block5_conv3_conv2d_readvariableop_resource:I
:model_1_vgg19_block5_conv3_biasadd_readvariableop_resource:	U
9model_1_vgg19_block5_conv4_conv2d_readvariableop_resource:I
:model_1_vgg19_block5_conv4_biasadd_readvariableop_resource:	A
.model_1_dense_1_matmul_readvariableop_resource:	@=
/model_1_dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢&model_1/dense_1/BiasAdd/ReadVariableOp¢(model_1/dense_1/BiasAdd_1/ReadVariableOp¢%model_1/dense_1/MatMul/ReadVariableOp¢'model_1/dense_1/MatMul_1/ReadVariableOp¢1model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp¢3model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp¢2model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp¢3model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp¢2model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp¢3model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp¢2model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp¢3model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp¢2model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp¢3model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp¢2model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp¢3model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp¢2model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp¢3model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp¢2model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp¢3model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp¢2model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp¢3model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp¢2model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp¢3model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp¢2model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp¢3model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp¢2model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp¢3model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp¢2model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp¢3model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp¢2model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp¢3model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp¢2model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp¢3model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp¢2model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp¢3model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp¢2model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp²
0model_1/vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ó
!model_1/vgg19/block1_conv1/Conv2DConv2Dinputs_08model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¨
1model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ð
"model_1/vgg19/block1_conv1/BiasAddBiasAdd*model_1/vgg19/block1_conv1/Conv2D:output:09model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
model_1/vgg19/block1_conv1/ReluRelu+model_1/vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@²
0model_1/vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ø
!model_1/vgg19/block1_conv2/Conv2DConv2D-model_1/vgg19/block1_conv1/Relu:activations:08model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¨
1model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ð
"model_1/vgg19/block1_conv2/BiasAddBiasAdd*model_1/vgg19/block1_conv2/Conv2D:output:09model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
model_1/vgg19/block1_conv2/ReluRelu+model_1/vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@È
!model_1/vgg19/block1_pool/MaxPoolMaxPool-model_1/vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
³
0model_1/vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ô
!model_1/vgg19/block2_conv1/Conv2DConv2D*model_1/vgg19/block1_pool/MaxPool:output:08model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
©
1model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block2_conv1/BiasAddBiasAdd*model_1/vgg19/block2_conv1/Conv2D:output:09model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
model_1/vgg19/block2_conv1/ReluRelu+model_1/vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd´
0model_1/vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block2_conv2/Conv2DConv2D-model_1/vgg19/block2_conv1/Relu:activations:08model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
©
1model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block2_conv2/BiasAddBiasAdd*model_1/vgg19/block2_conv2/Conv2D:output:09model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
model_1/vgg19/block2_conv2/ReluRelu+model_1/vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÉ
!model_1/vgg19/block2_pool/MaxPoolMaxPool-model_1/vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
´
0model_1/vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ô
!model_1/vgg19/block3_conv1/Conv2DConv2D*model_1/vgg19/block2_pool/MaxPool:output:08model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
©
1model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block3_conv1/BiasAddBiasAdd*model_1/vgg19/block3_conv1/Conv2D:output:09model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
model_1/vgg19/block3_conv1/ReluRelu+model_1/vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22´
0model_1/vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block3_conv2/Conv2DConv2D-model_1/vgg19/block3_conv1/Relu:activations:08model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
©
1model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block3_conv2/BiasAddBiasAdd*model_1/vgg19/block3_conv2/Conv2D:output:09model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
model_1/vgg19/block3_conv2/ReluRelu+model_1/vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22´
0model_1/vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block3_conv3/Conv2DConv2D-model_1/vgg19/block3_conv2/Relu:activations:08model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
©
1model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block3_conv3/BiasAddBiasAdd*model_1/vgg19/block3_conv3/Conv2D:output:09model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
model_1/vgg19/block3_conv3/ReluRelu+model_1/vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22´
0model_1/vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block3_conv4/Conv2DConv2D-model_1/vgg19/block3_conv3/Relu:activations:08model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
©
1model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block3_conv4/BiasAddBiasAdd*model_1/vgg19/block3_conv4/Conv2D:output:09model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
model_1/vgg19/block3_conv4/ReluRelu+model_1/vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22É
!model_1/vgg19/block3_pool/MaxPoolMaxPool-model_1/vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
´
0model_1/vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ô
!model_1/vgg19/block4_conv1/Conv2DConv2D*model_1/vgg19/block3_pool/MaxPool:output:08model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block4_conv1/BiasAddBiasAdd*model_1/vgg19/block4_conv1/Conv2D:output:09model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block4_conv1/ReluRelu+model_1/vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block4_conv2/Conv2DConv2D-model_1/vgg19/block4_conv1/Relu:activations:08model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block4_conv2/BiasAddBiasAdd*model_1/vgg19/block4_conv2/Conv2D:output:09model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block4_conv2/ReluRelu+model_1/vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block4_conv3/Conv2DConv2D-model_1/vgg19/block4_conv2/Relu:activations:08model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block4_conv3/BiasAddBiasAdd*model_1/vgg19/block4_conv3/Conv2D:output:09model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block4_conv3/ReluRelu+model_1/vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block4_conv4/Conv2DConv2D-model_1/vgg19/block4_conv3/Relu:activations:08model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block4_conv4/BiasAddBiasAdd*model_1/vgg19/block4_conv4/Conv2D:output:09model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block4_conv4/ReluRelu+model_1/vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
!model_1/vgg19/block4_pool/MaxPoolMaxPool-model_1/vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
´
0model_1/vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ô
!model_1/vgg19/block5_conv1/Conv2DConv2D*model_1/vgg19/block4_pool/MaxPool:output:08model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block5_conv1/BiasAddBiasAdd*model_1/vgg19/block5_conv1/Conv2D:output:09model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block5_conv1/ReluRelu+model_1/vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block5_conv2/Conv2DConv2D-model_1/vgg19/block5_conv1/Relu:activations:08model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block5_conv2/BiasAddBiasAdd*model_1/vgg19/block5_conv2/Conv2D:output:09model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block5_conv2/ReluRelu+model_1/vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block5_conv3/Conv2DConv2D-model_1/vgg19/block5_conv2/Relu:activations:08model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block5_conv3/BiasAddBiasAdd*model_1/vgg19/block5_conv3/Conv2D:output:09model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block5_conv3/ReluRelu+model_1/vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block5_conv4/Conv2DConv2D-model_1/vgg19/block5_conv3/Relu:activations:08model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block5_conv4/BiasAddBiasAdd*model_1/vgg19/block5_conv4/Conv2D:output:09model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block5_conv4/ReluRelu+model_1/vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
!model_1/vgg19/block5_pool/MaxPoolMaxPool-model_1/vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

9model_1/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ò
'model_1/global_average_pooling2d_1/MeanMean*model_1/vgg19/block5_pool/MaxPool:output:0Bmodel_1/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0³
model_1/dense_1/MatMulMatMul0model_1/global_average_pooling2d_1/Mean:output:0-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@´
2model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0×
#model_1/vgg19/block1_conv1/Conv2D_1Conv2Dinputs_1:model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
ª
3model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
$model_1/vgg19/block1_conv1/BiasAdd_1BiasAdd,model_1/vgg19/block1_conv1/Conv2D_1:output:0;model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
!model_1/vgg19/block1_conv1/Relu_1Relu-model_1/vgg19/block1_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@´
2model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0þ
#model_1/vgg19/block1_conv2/Conv2D_1Conv2D/model_1/vgg19/block1_conv1/Relu_1:activations:0:model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
ª
3model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
$model_1/vgg19/block1_conv2/BiasAdd_1BiasAdd,model_1/vgg19/block1_conv2/Conv2D_1:output:0;model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
!model_1/vgg19/block1_conv2/Relu_1Relu-model_1/vgg19/block1_conv2/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Ì
#model_1/vgg19/block1_pool/MaxPool_1MaxPool/model_1/vgg19/block1_conv2/Relu_1:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
µ
2model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ú
#model_1/vgg19/block2_conv1/Conv2D_1Conv2D,model_1/vgg19/block1_pool/MaxPool_1:output:0:model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
«
3model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block2_conv1/BiasAdd_1BiasAdd,model_1/vgg19/block2_conv1/Conv2D_1:output:0;model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!model_1/vgg19/block2_conv1/Relu_1Relu-model_1/vgg19/block2_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¶
2model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block2_conv2/Conv2D_1Conv2D/model_1/vgg19/block2_conv1/Relu_1:activations:0:model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
«
3model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block2_conv2/BiasAdd_1BiasAdd,model_1/vgg19/block2_conv2/Conv2D_1:output:0;model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!model_1/vgg19/block2_conv2/Relu_1Relu-model_1/vgg19/block2_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÍ
#model_1/vgg19/block2_pool/MaxPool_1MaxPool/model_1/vgg19/block2_conv2/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
¶
2model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
#model_1/vgg19/block3_conv1/Conv2D_1Conv2D,model_1/vgg19/block2_pool/MaxPool_1:output:0:model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
3model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block3_conv1/BiasAdd_1BiasAdd,model_1/vgg19/block3_conv1/Conv2D_1:output:0;model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!model_1/vgg19/block3_conv1/Relu_1Relu-model_1/vgg19/block3_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
2model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block3_conv2/Conv2D_1Conv2D/model_1/vgg19/block3_conv1/Relu_1:activations:0:model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
3model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block3_conv2/BiasAdd_1BiasAdd,model_1/vgg19/block3_conv2/Conv2D_1:output:0;model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!model_1/vgg19/block3_conv2/Relu_1Relu-model_1/vgg19/block3_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
2model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block3_conv3/Conv2D_1Conv2D/model_1/vgg19/block3_conv2/Relu_1:activations:0:model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
3model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block3_conv3/BiasAdd_1BiasAdd,model_1/vgg19/block3_conv3/Conv2D_1:output:0;model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!model_1/vgg19/block3_conv3/Relu_1Relu-model_1/vgg19/block3_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
2model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block3_conv4/Conv2D_1Conv2D/model_1/vgg19/block3_conv3/Relu_1:activations:0:model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
3model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block3_conv4/BiasAdd_1BiasAdd,model_1/vgg19/block3_conv4/Conv2D_1:output:0;model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!model_1/vgg19/block3_conv4/Relu_1Relu-model_1/vgg19/block3_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Í
#model_1/vgg19/block3_pool/MaxPool_1MaxPool/model_1/vgg19/block3_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¶
2model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
#model_1/vgg19/block4_conv1/Conv2D_1Conv2D,model_1/vgg19/block3_pool/MaxPool_1:output:0:model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block4_conv1/BiasAdd_1BiasAdd,model_1/vgg19/block4_conv1/Conv2D_1:output:0;model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block4_conv1/Relu_1Relu-model_1/vgg19/block4_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block4_conv2/Conv2D_1Conv2D/model_1/vgg19/block4_conv1/Relu_1:activations:0:model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block4_conv2/BiasAdd_1BiasAdd,model_1/vgg19/block4_conv2/Conv2D_1:output:0;model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block4_conv2/Relu_1Relu-model_1/vgg19/block4_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block4_conv3/Conv2D_1Conv2D/model_1/vgg19/block4_conv2/Relu_1:activations:0:model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block4_conv3/BiasAdd_1BiasAdd,model_1/vgg19/block4_conv3/Conv2D_1:output:0;model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block4_conv3/Relu_1Relu-model_1/vgg19/block4_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block4_conv4/Conv2D_1Conv2D/model_1/vgg19/block4_conv3/Relu_1:activations:0:model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block4_conv4/BiasAdd_1BiasAdd,model_1/vgg19/block4_conv4/Conv2D_1:output:0;model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block4_conv4/Relu_1Relu-model_1/vgg19/block4_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
#model_1/vgg19/block4_pool/MaxPool_1MaxPool/model_1/vgg19/block4_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¶
2model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
#model_1/vgg19/block5_conv1/Conv2D_1Conv2D,model_1/vgg19/block4_pool/MaxPool_1:output:0:model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block5_conv1/BiasAdd_1BiasAdd,model_1/vgg19/block5_conv1/Conv2D_1:output:0;model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block5_conv1/Relu_1Relu-model_1/vgg19/block5_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block5_conv2/Conv2D_1Conv2D/model_1/vgg19/block5_conv1/Relu_1:activations:0:model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block5_conv2/BiasAdd_1BiasAdd,model_1/vgg19/block5_conv2/Conv2D_1:output:0;model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block5_conv2/Relu_1Relu-model_1/vgg19/block5_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block5_conv3/Conv2D_1Conv2D/model_1/vgg19/block5_conv2/Relu_1:activations:0:model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block5_conv3/BiasAdd_1BiasAdd,model_1/vgg19/block5_conv3/Conv2D_1:output:0;model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block5_conv3/Relu_1Relu-model_1/vgg19/block5_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block5_conv4/Conv2D_1Conv2D/model_1/vgg19/block5_conv3/Relu_1:activations:0:model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block5_conv4/BiasAdd_1BiasAdd,model_1/vgg19/block5_conv4/Conv2D_1:output:0;model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block5_conv4/Relu_1Relu-model_1/vgg19/block5_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
#model_1/vgg19/block5_pool/MaxPool_1MaxPool/model_1/vgg19/block5_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

;model_1/global_average_pooling2d_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ø
)model_1/global_average_pooling2d_1/Mean_1Mean,model_1/vgg19/block5_pool/MaxPool_1:output:0Dmodel_1/global_average_pooling2d_1/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¹
model_1/dense_1/MatMul_1MatMul2model_1/global_average_pooling2d_1/Mean_1:output:0/model_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(model_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¬
model_1/dense_1/BiasAdd_1BiasAdd"model_1/dense_1/MatMul_1:product:00model_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

lambda/subSub model_1/dense_1/BiasAdd:output:0"model_1/dense_1/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
lambda/SquareSquarelambda/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
lambda/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :

lambda/SumSumlambda/Square:y:0%lambda/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(U
lambda/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3{
lambda/MaximumMaximumlambda/Sum:output:0lambda/Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
lambda/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
lambda/Maximum_1Maximumlambda/Maximum:z:0lambda/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lambda/SqrtSqrtlambda/Maximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMullambda/Sqrt:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp'^model_1/dense_1/BiasAdd/ReadVariableOp)^model_1/dense_1/BiasAdd_1/ReadVariableOp&^model_1/dense_1/MatMul/ReadVariableOp(^model_1/dense_1/MatMul_1/ReadVariableOp2^model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp4^model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp3^model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp2^model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp4^model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp3^model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp2^model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp4^model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp3^model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp2^model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp4^model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp3^model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp2^model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp4^model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp3^model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp2^model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp4^model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp3^model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp2^model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp4^model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp3^model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp2^model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp4^model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp3^model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp2^model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp4^model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp3^model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp2^model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp4^model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp3^model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp2^model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp4^model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp3^model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp2^model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp4^model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp3^model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp2^model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp4^model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp3^model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp2^model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp4^model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp3^model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp2^model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp4^model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp3^model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp2^model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp4^model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp3^model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2P
&model_1/dense_1/BiasAdd/ReadVariableOp&model_1/dense_1/BiasAdd/ReadVariableOp2T
(model_1/dense_1/BiasAdd_1/ReadVariableOp(model_1/dense_1/BiasAdd_1/ReadVariableOp2N
%model_1/dense_1/MatMul/ReadVariableOp%model_1/dense_1/MatMul/ReadVariableOp2R
'model_1/dense_1/MatMul_1/ReadVariableOp'model_1/dense_1/MatMul_1/ReadVariableOp2f
1model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp1model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp3model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp0model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp2h
2model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp2model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp1model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp3model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp0model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp2h
2model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp2model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp1model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp3model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp0model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp2h
2model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp2model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp1model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp3model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp0model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp2h
2model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp2model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp1model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp3model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp0model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp2h
2model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp2model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp1model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp3model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp0model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp2h
2model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp2model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp1model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp3model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp0model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp2h
2model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp2model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp1model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp3model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp0model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp2h
2model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp2model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp1model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp3model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp0model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp2h
2model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp2model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp1model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp3model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp0model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp2h
2model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp2model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp1model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp3model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp0model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp2h
2model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp2model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp1model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp3model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp0model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp2h
2model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp2model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp1model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp3model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp0model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp2h
2model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp2model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp1model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp3model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp0model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp2h
2model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp2model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp1model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp3model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp0model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp2h
2model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp2model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp1model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp3model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp0model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp2h
2model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp2model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/1


I__inference_block4_conv4_layer_call_and_return_conditional_losses_1038847

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block2_conv1_layer_call_and_return_conditional_losses_1034588

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@
 
_user_specified_nameinputs


I__inference_block5_conv2_layer_call_and_return_conditional_losses_1034778

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
H__inference_block4_pool_layer_call_and_return_conditional_losses_1038857

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block3_conv1_layer_call_and_return_conditional_losses_1034623

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs


I__inference_block3_conv4_layer_call_and_return_conditional_losses_1034674

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs

d
H__inference_block3_pool_layer_call_and_return_conditional_losses_1034508

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
ê
B__inference_vgg19_layer_call_and_return_conditional_losses_1038547

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@;
,block2_conv1_biasadd_readvariableop_resource:	G
+block2_conv2_conv2d_readvariableop_resource:;
,block2_conv2_biasadd_readvariableop_resource:	G
+block3_conv1_conv2d_readvariableop_resource:;
,block3_conv1_biasadd_readvariableop_resource:	G
+block3_conv2_conv2d_readvariableop_resource:;
,block3_conv2_biasadd_readvariableop_resource:	G
+block3_conv3_conv2d_readvariableop_resource:;
,block3_conv3_biasadd_readvariableop_resource:	G
+block3_conv4_conv2d_readvariableop_resource:;
,block3_conv4_biasadd_readvariableop_resource:	G
+block4_conv1_conv2d_readvariableop_resource:;
,block4_conv1_biasadd_readvariableop_resource:	G
+block4_conv2_conv2d_readvariableop_resource:;
,block4_conv2_biasadd_readvariableop_resource:	G
+block4_conv3_conv2d_readvariableop_resource:;
,block4_conv3_biasadd_readvariableop_resource:	G
+block4_conv4_conv2d_readvariableop_resource:;
,block4_conv4_biasadd_readvariableop_resource:	G
+block5_conv1_conv2d_readvariableop_resource:;
,block5_conv1_biasadd_readvariableop_resource:	G
+block5_conv2_conv2d_readvariableop_resource:;
,block5_conv2_biasadd_readvariableop_resource:	G
+block5_conv3_conv2d_readvariableop_resource:;
,block5_conv3_biasadd_readvariableop_resource:	G
+block5_conv4_conv2d_readvariableop_resource:;
,block5_conv4_biasadd_readvariableop_resource:	
identity¢#block1_conv1/BiasAdd/ReadVariableOp¢"block1_conv1/Conv2D/ReadVariableOp¢#block1_conv2/BiasAdd/ReadVariableOp¢"block1_conv2/Conv2D/ReadVariableOp¢#block2_conv1/BiasAdd/ReadVariableOp¢"block2_conv1/Conv2D/ReadVariableOp¢#block2_conv2/BiasAdd/ReadVariableOp¢"block2_conv2/Conv2D/ReadVariableOp¢#block3_conv1/BiasAdd/ReadVariableOp¢"block3_conv1/Conv2D/ReadVariableOp¢#block3_conv2/BiasAdd/ReadVariableOp¢"block3_conv2/Conv2D/ReadVariableOp¢#block3_conv3/BiasAdd/ReadVariableOp¢"block3_conv3/Conv2D/ReadVariableOp¢#block3_conv4/BiasAdd/ReadVariableOp¢"block3_conv4/Conv2D/ReadVariableOp¢#block4_conv1/BiasAdd/ReadVariableOp¢"block4_conv1/Conv2D/ReadVariableOp¢#block4_conv2/BiasAdd/ReadVariableOp¢"block4_conv2/Conv2D/ReadVariableOp¢#block4_conv3/BiasAdd/ReadVariableOp¢"block4_conv3/Conv2D/ReadVariableOp¢#block4_conv4/BiasAdd/ReadVariableOp¢"block4_conv4/Conv2D/ReadVariableOp¢#block5_conv1/BiasAdd/ReadVariableOp¢"block5_conv1/Conv2D/ReadVariableOp¢#block5_conv2/BiasAdd/ReadVariableOp¢"block5_conv2/Conv2D/ReadVariableOp¢#block5_conv3/BiasAdd/ReadVariableOp¢"block5_conv3/Conv2D/ReadVariableOp¢#block5_conv4/BiasAdd/ReadVariableOp¢"block5_conv4/Conv2D/ReadVariableOp
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0µ
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Î
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides

#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¬
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides

"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ê
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdds
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides

#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdds
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd­
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides

"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
"block3_conv4/Conv2D/ReadVariableOpReadVariableOp+block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block3_conv4/Conv2DConv2Dblock3_conv3/Relu:activations:0*block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

#block3_conv4/BiasAdd/ReadVariableOpReadVariableOp,block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block3_conv4/BiasAddBiasAddblock3_conv4/Conv2D:output:0+block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22s
block3_conv4/ReluRelublock3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22­
block3_pool/MaxPoolMaxPoolblock3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block4_conv4/Conv2D/ReadVariableOpReadVariableOp+block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block4_conv4/Conv2DConv2Dblock4_conv3/Relu:activations:0*block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block4_conv4/BiasAdd/ReadVariableOpReadVariableOp,block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block4_conv4/BiasAddBiasAddblock4_conv4/Conv2D:output:0+block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block4_conv4/ReluRelublock4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block4_pool/MaxPoolMaxPoolblock4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ê
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"block5_conv4/Conv2D/ReadVariableOpReadVariableOp+block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Í
block5_conv4/Conv2DConv2Dblock5_conv3/Relu:activations:0*block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

#block5_conv4/BiasAdd/ReadVariableOpReadVariableOp,block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¥
block5_conv4/BiasAddBiasAddblock5_conv4/Conv2D:output:0+block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
block5_conv4/ReluRelublock5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
block5_pool/MaxPoolMaxPoolblock5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿö	
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block3_conv4/BiasAdd/ReadVariableOp#^block3_conv4/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block4_conv4/BiasAdd/ReadVariableOp#^block4_conv4/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp$^block5_conv4/BiasAdd/ReadVariableOp#^block5_conv4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block3_conv4/BiasAdd/ReadVariableOp#block3_conv4/BiasAdd/ReadVariableOp2H
"block3_conv4/Conv2D/ReadVariableOp"block3_conv4/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block4_conv4/BiasAdd/ReadVariableOp#block4_conv4/BiasAdd/ReadVariableOp2H
"block4_conv4/Conv2D/ReadVariableOp"block4_conv4/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp2J
#block5_conv4/BiasAdd/ReadVariableOp#block5_conv4/BiasAdd/ReadVariableOp2H
"block5_conv4/Conv2D/ReadVariableOp"block5_conv4/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs


I__inference_block1_conv2_layer_call_and_return_conditional_losses_1034570

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÈÈ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
 
_user_specified_nameinputs
å
ÿ	
)__inference_model_2_layer_call_fn_1037149
inputs_0
inputs_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	

unknown_31:	@

unknown_32:@

unknown_33:

unknown_34:
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_1036604o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/1


I__inference_block4_conv2_layer_call_and_return_conditional_losses_1034709

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õa
©
 __inference__traced_save_1039130
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block3_conv4_kernel_read_readvariableop0
,savev2_block3_conv4_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block4_conv4_kernel_read_readvariableop0
,savev2_block4_conv4_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop2
.savev2_block5_conv4_kernel_read_readvariableop0
,savev2_block5_conv4_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ý
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*¦
valueB6B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÙ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Í
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block3_conv4_kernel_read_readvariableop,savev2_block3_conv4_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block4_conv4_kernel_read_readvariableop,savev2_block4_conv4_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop.savev2_block5_conv4_kernel_read_readvariableop,savev2_block5_conv4_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *D
dtypes:
826	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*·
_input_shapes¥
¢: ::: : : : : :@:@:@@:@:@::::::::::::::::::::::::::::	@:@: : : : :::	@:@:::	@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 	

_output_shapes
:@:,
(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::. *
(
_output_shapes
::!!

_output_shapes	
::."*
(
_output_shapes
::!#

_output_shapes	
::.$*
(
_output_shapes
::!%

_output_shapes	
::.&*
(
_output_shapes
::!'

_output_shapes	
::%(!

_output_shapes
:	@: )

_output_shapes
:@:*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :$. 

_output_shapes

:: /

_output_shapes
::%0!

_output_shapes
:	@: 1

_output_shapes
:@:$2 

_output_shapes

:: 3

_output_shapes
::%4!

_output_shapes
:	@: 5

_output_shapes
:@:6

_output_shapes
: 
µ
I
-__inference_block2_pool_layer_call_fn_1038672

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_block2_pool_layer_call_and_return_conditional_losses_1034496
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
ý
'__inference_vgg19_layer_call_fn_1035343
input_6!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_vgg19_layer_call_and_return_conditional_losses_1035207x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_6
êÚ
²9
"__inference__wrapped_model_1034475
input_3
input_4[
Amodel_2_model_1_vgg19_block1_conv1_conv2d_readvariableop_resource:@P
Bmodel_2_model_1_vgg19_block1_conv1_biasadd_readvariableop_resource:@[
Amodel_2_model_1_vgg19_block1_conv2_conv2d_readvariableop_resource:@@P
Bmodel_2_model_1_vgg19_block1_conv2_biasadd_readvariableop_resource:@\
Amodel_2_model_1_vgg19_block2_conv1_conv2d_readvariableop_resource:@Q
Bmodel_2_model_1_vgg19_block2_conv1_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block2_conv2_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block2_conv2_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block3_conv1_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block3_conv1_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block3_conv2_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block3_conv2_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block3_conv3_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block3_conv3_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block3_conv4_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block3_conv4_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block4_conv1_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block4_conv1_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block4_conv2_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block4_conv2_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block4_conv3_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block4_conv3_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block4_conv4_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block4_conv4_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block5_conv1_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block5_conv1_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block5_conv2_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block5_conv2_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block5_conv3_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block5_conv3_biasadd_readvariableop_resource:	]
Amodel_2_model_1_vgg19_block5_conv4_conv2d_readvariableop_resource:Q
Bmodel_2_model_1_vgg19_block5_conv4_biasadd_readvariableop_resource:	I
6model_2_model_1_dense_1_matmul_readvariableop_resource:	@E
7model_2_model_1_dense_1_biasadd_readvariableop_resource:@@
.model_2_dense_2_matmul_readvariableop_resource:=
/model_2_dense_2_biasadd_readvariableop_resource:
identity¢&model_2/dense_2/BiasAdd/ReadVariableOp¢%model_2/dense_2/MatMul/ReadVariableOp¢.model_2/model_1/dense_1/BiasAdd/ReadVariableOp¢0model_2/model_1/dense_1/BiasAdd_1/ReadVariableOp¢-model_2/model_1/dense_1/MatMul/ReadVariableOp¢/model_2/model_1/dense_1/MatMul_1/ReadVariableOp¢9model_2/model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp¢9model_2/model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp¢;model_2/model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp¢8model_2/model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp¢:model_2/model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOpÂ
8model_2/model_1/vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0â
)model_2/model_1/vgg19/block1_conv1/Conv2DConv2Dinput_3@model_2/model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¸
9model_2/model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0è
*model_2/model_1/vgg19/block1_conv1/BiasAddBiasAdd2model_2/model_1/vgg19/block1_conv1/Conv2D:output:0Amodel_2/model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@ 
'model_2/model_1/vgg19/block1_conv1/ReluRelu3model_2/model_1/vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Â
8model_2/model_1/vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
)model_2/model_1/vgg19/block1_conv2/Conv2DConv2D5model_2/model_1/vgg19/block1_conv1/Relu:activations:0@model_2/model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¸
9model_2/model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0è
*model_2/model_1/vgg19/block1_conv2/BiasAddBiasAdd2model_2/model_1/vgg19/block1_conv2/Conv2D:output:0Amodel_2/model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@ 
'model_2/model_1/vgg19/block1_conv2/ReluRelu3model_2/model_1/vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Ø
)model_2/model_1/vgg19/block1_pool/MaxPoolMaxPool5model_2/model_1/vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
Ã
8model_2/model_1/vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
)model_2/model_1/vgg19/block2_conv1/Conv2DConv2D2model_2/model_1/vgg19/block1_pool/MaxPool:output:0@model_2/model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block2_conv1/BiasAddBiasAdd2model_2/model_1/vgg19/block2_conv1/Conv2D:output:0Amodel_2/model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
'model_2/model_1/vgg19/block2_conv1/ReluRelu3model_2/model_1/vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÄ
8model_2/model_1/vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block2_conv2/Conv2DConv2D5model_2/model_1/vgg19/block2_conv1/Relu:activations:0@model_2/model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block2_conv2/BiasAddBiasAdd2model_2/model_1/vgg19/block2_conv2/Conv2D:output:0Amodel_2/model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
'model_2/model_1/vgg19/block2_conv2/ReluRelu3model_2/model_1/vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÙ
)model_2/model_1/vgg19/block2_pool/MaxPoolMaxPool5model_2/model_1/vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
Ä
8model_2/model_1/vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block3_conv1/Conv2DConv2D2model_2/model_1/vgg19/block2_pool/MaxPool:output:0@model_2/model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block3_conv1/BiasAddBiasAdd2model_2/model_1/vgg19/block3_conv1/Conv2D:output:0Amodel_2/model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
'model_2/model_1/vgg19/block3_conv1/ReluRelu3model_2/model_1/vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ä
8model_2/model_1/vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block3_conv2/Conv2DConv2D5model_2/model_1/vgg19/block3_conv1/Relu:activations:0@model_2/model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block3_conv2/BiasAddBiasAdd2model_2/model_1/vgg19/block3_conv2/Conv2D:output:0Amodel_2/model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
'model_2/model_1/vgg19/block3_conv2/ReluRelu3model_2/model_1/vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ä
8model_2/model_1/vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block3_conv3/Conv2DConv2D5model_2/model_1/vgg19/block3_conv2/Relu:activations:0@model_2/model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block3_conv3/BiasAddBiasAdd2model_2/model_1/vgg19/block3_conv3/Conv2D:output:0Amodel_2/model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
'model_2/model_1/vgg19/block3_conv3/ReluRelu3model_2/model_1/vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ä
8model_2/model_1/vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block3_conv4/Conv2DConv2D5model_2/model_1/vgg19/block3_conv3/Relu:activations:0@model_2/model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block3_conv4/BiasAddBiasAdd2model_2/model_1/vgg19/block3_conv4/Conv2D:output:0Amodel_2/model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
'model_2/model_1/vgg19/block3_conv4/ReluRelu3model_2/model_1/vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ù
)model_2/model_1/vgg19/block3_pool/MaxPoolMaxPool5model_2/model_1/vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
Ä
8model_2/model_1/vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block4_conv1/Conv2DConv2D2model_2/model_1/vgg19/block3_pool/MaxPool:output:0@model_2/model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block4_conv1/BiasAddBiasAdd2model_2/model_1/vgg19/block4_conv1/Conv2D:output:0Amodel_2/model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_2/model_1/vgg19/block4_conv1/ReluRelu3model_2/model_1/vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
8model_2/model_1/vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block4_conv2/Conv2DConv2D5model_2/model_1/vgg19/block4_conv1/Relu:activations:0@model_2/model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block4_conv2/BiasAddBiasAdd2model_2/model_1/vgg19/block4_conv2/Conv2D:output:0Amodel_2/model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_2/model_1/vgg19/block4_conv2/ReluRelu3model_2/model_1/vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
8model_2/model_1/vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block4_conv3/Conv2DConv2D5model_2/model_1/vgg19/block4_conv2/Relu:activations:0@model_2/model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block4_conv3/BiasAddBiasAdd2model_2/model_1/vgg19/block4_conv3/Conv2D:output:0Amodel_2/model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_2/model_1/vgg19/block4_conv3/ReluRelu3model_2/model_1/vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
8model_2/model_1/vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block4_conv4/Conv2DConv2D5model_2/model_1/vgg19/block4_conv3/Relu:activations:0@model_2/model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block4_conv4/BiasAddBiasAdd2model_2/model_1/vgg19/block4_conv4/Conv2D:output:0Amodel_2/model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_2/model_1/vgg19/block4_conv4/ReluRelu3model_2/model_1/vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
)model_2/model_1/vgg19/block4_pool/MaxPoolMaxPool5model_2/model_1/vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
Ä
8model_2/model_1/vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block5_conv1/Conv2DConv2D2model_2/model_1/vgg19/block4_pool/MaxPool:output:0@model_2/model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block5_conv1/BiasAddBiasAdd2model_2/model_1/vgg19/block5_conv1/Conv2D:output:0Amodel_2/model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_2/model_1/vgg19/block5_conv1/ReluRelu3model_2/model_1/vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
8model_2/model_1/vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block5_conv2/Conv2DConv2D5model_2/model_1/vgg19/block5_conv1/Relu:activations:0@model_2/model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block5_conv2/BiasAddBiasAdd2model_2/model_1/vgg19/block5_conv2/Conv2D:output:0Amodel_2/model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_2/model_1/vgg19/block5_conv2/ReluRelu3model_2/model_1/vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
8model_2/model_1/vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block5_conv3/Conv2DConv2D5model_2/model_1/vgg19/block5_conv2/Relu:activations:0@model_2/model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block5_conv3/BiasAddBiasAdd2model_2/model_1/vgg19/block5_conv3/Conv2D:output:0Amodel_2/model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_2/model_1/vgg19/block5_conv3/ReluRelu3model_2/model_1/vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
8model_2/model_1/vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)model_2/model_1/vgg19/block5_conv4/Conv2DConv2D5model_2/model_1/vgg19/block5_conv3/Relu:activations:0@model_2/model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¹
9model_2/model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ç
*model_2/model_1/vgg19/block5_conv4/BiasAddBiasAdd2model_2/model_1/vgg19/block5_conv4/Conv2D:output:0Amodel_2/model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_2/model_1/vgg19/block5_conv4/ReluRelu3model_2/model_1/vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
)model_2/model_1/vgg19/block5_pool/MaxPoolMaxPool5model_2/model_1/vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

Amodel_2/model_1/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ê
/model_2/model_1/global_average_pooling2d_1/MeanMean2model_2/model_1/vgg19/block5_pool/MaxPool:output:0Jmodel_2/model_1/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-model_2/model_1/dense_1/MatMul/ReadVariableOpReadVariableOp6model_2_model_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0Ë
model_2/model_1/dense_1/MatMulMatMul8model_2/model_1/global_average_pooling2d_1/Mean:output:05model_2/model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¢
.model_2/model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp7model_2_model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¾
model_2/model_1/dense_1/BiasAddBiasAdd(model_2/model_1/dense_1/MatMul:product:06model_2/model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ä
:model_2/model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0æ
+model_2/model_1/vgg19/block1_conv1/Conv2D_1Conv2Dinput_4Bmodel_2/model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
º
;model_2/model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
,model_2/model_1/vgg19/block1_conv1/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block1_conv1/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¤
)model_2/model_1/vgg19/block1_conv1/Relu_1Relu5model_2/model_1/vgg19/block1_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Ä
:model_2/model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
+model_2/model_1/vgg19/block1_conv2/Conv2D_1Conv2D7model_2/model_1/vgg19/block1_conv1/Relu_1:activations:0Bmodel_2/model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
º
;model_2/model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0î
,model_2/model_1/vgg19/block1_conv2/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block1_conv2/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@¤
)model_2/model_1/vgg19/block1_conv2/Relu_1Relu5model_2/model_1/vgg19/block1_conv2/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Ü
+model_2/model_1/vgg19/block1_pool/MaxPool_1MaxPool7model_2/model_1/vgg19/block1_conv2/Relu_1:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
Å
:model_2/model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
+model_2/model_1/vgg19/block2_conv1/Conv2D_1Conv2D4model_2/model_1/vgg19/block1_pool/MaxPool_1:output:0Bmodel_2/model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block2_conv1/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block2_conv1/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd£
)model_2/model_1/vgg19/block2_conv1/Relu_1Relu5model_2/model_1/vgg19/block2_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÆ
:model_2/model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block2_conv2/Conv2D_1Conv2D7model_2/model_1/vgg19/block2_conv1/Relu_1:activations:0Bmodel_2/model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block2_conv2/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block2_conv2/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd£
)model_2/model_1/vgg19/block2_conv2/Relu_1Relu5model_2/model_1/vgg19/block2_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÝ
+model_2/model_1/vgg19/block2_pool/MaxPool_1MaxPool7model_2/model_1/vgg19/block2_conv2/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
Æ
:model_2/model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block3_conv1/Conv2D_1Conv2D4model_2/model_1/vgg19/block2_pool/MaxPool_1:output:0Bmodel_2/model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block3_conv1/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block3_conv1/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22£
)model_2/model_1/vgg19/block3_conv1/Relu_1Relu5model_2/model_1/vgg19/block3_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Æ
:model_2/model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block3_conv2/Conv2D_1Conv2D7model_2/model_1/vgg19/block3_conv1/Relu_1:activations:0Bmodel_2/model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block3_conv2/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block3_conv2/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22£
)model_2/model_1/vgg19/block3_conv2/Relu_1Relu5model_2/model_1/vgg19/block3_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Æ
:model_2/model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block3_conv3/Conv2D_1Conv2D7model_2/model_1/vgg19/block3_conv2/Relu_1:activations:0Bmodel_2/model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block3_conv3/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block3_conv3/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22£
)model_2/model_1/vgg19/block3_conv3/Relu_1Relu5model_2/model_1/vgg19/block3_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Æ
:model_2/model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block3_conv4/Conv2D_1Conv2D7model_2/model_1/vgg19/block3_conv3/Relu_1:activations:0Bmodel_2/model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block3_conv4/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block3_conv4/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22£
)model_2/model_1/vgg19/block3_conv4/Relu_1Relu5model_2/model_1/vgg19/block3_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ý
+model_2/model_1/vgg19/block3_pool/MaxPool_1MaxPool7model_2/model_1/vgg19/block3_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
Æ
:model_2/model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block4_conv1/Conv2D_1Conv2D4model_2/model_1/vgg19/block3_pool/MaxPool_1:output:0Bmodel_2/model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block4_conv1/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block4_conv1/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_2/model_1/vgg19/block4_conv1/Relu_1Relu5model_2/model_1/vgg19/block4_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
:model_2/model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block4_conv2/Conv2D_1Conv2D7model_2/model_1/vgg19/block4_conv1/Relu_1:activations:0Bmodel_2/model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block4_conv2/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block4_conv2/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_2/model_1/vgg19/block4_conv2/Relu_1Relu5model_2/model_1/vgg19/block4_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
:model_2/model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block4_conv3/Conv2D_1Conv2D7model_2/model_1/vgg19/block4_conv2/Relu_1:activations:0Bmodel_2/model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block4_conv3/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block4_conv3/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_2/model_1/vgg19/block4_conv3/Relu_1Relu5model_2/model_1/vgg19/block4_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
:model_2/model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block4_conv4/Conv2D_1Conv2D7model_2/model_1/vgg19/block4_conv3/Relu_1:activations:0Bmodel_2/model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block4_conv4/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block4_conv4/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_2/model_1/vgg19/block4_conv4/Relu_1Relu5model_2/model_1/vgg19/block4_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
+model_2/model_1/vgg19/block4_pool/MaxPool_1MaxPool7model_2/model_1/vgg19/block4_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
Æ
:model_2/model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block5_conv1/Conv2D_1Conv2D4model_2/model_1/vgg19/block4_pool/MaxPool_1:output:0Bmodel_2/model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block5_conv1/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block5_conv1/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_2/model_1/vgg19/block5_conv1/Relu_1Relu5model_2/model_1/vgg19/block5_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
:model_2/model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block5_conv2/Conv2D_1Conv2D7model_2/model_1/vgg19/block5_conv1/Relu_1:activations:0Bmodel_2/model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block5_conv2/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block5_conv2/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_2/model_1/vgg19/block5_conv2/Relu_1Relu5model_2/model_1/vgg19/block5_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
:model_2/model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block5_conv3/Conv2D_1Conv2D7model_2/model_1/vgg19/block5_conv2/Relu_1:activations:0Bmodel_2/model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block5_conv3/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block5_conv3/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_2/model_1/vgg19/block5_conv3/Relu_1Relu5model_2/model_1/vgg19/block5_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
:model_2/model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOpReadVariableOpAmodel_2_model_1_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
+model_2/model_1/vgg19/block5_conv4/Conv2D_1Conv2D7model_2/model_1/vgg19/block5_conv3/Relu_1:activations:0Bmodel_2/model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
»
;model_2/model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOpReadVariableOpBmodel_2_model_1_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0í
,model_2/model_1/vgg19/block5_conv4/BiasAdd_1BiasAdd4model_2/model_1/vgg19/block5_conv4/Conv2D_1:output:0Cmodel_2/model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
)model_2/model_1/vgg19/block5_conv4/Relu_1Relu5model_2/model_1/vgg19/block5_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
+model_2/model_1/vgg19/block5_pool/MaxPool_1MaxPool7model_2/model_1/vgg19/block5_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

Cmodel_2/model_1/global_average_pooling2d_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ð
1model_2/model_1/global_average_pooling2d_1/Mean_1Mean4model_2/model_1/vgg19/block5_pool/MaxPool_1:output:0Lmodel_2/model_1/global_average_pooling2d_1/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
/model_2/model_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp6model_2_model_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0Ñ
 model_2/model_1/dense_1/MatMul_1MatMul:model_2/model_1/global_average_pooling2d_1/Mean_1:output:07model_2/model_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¤
0model_2/model_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp7model_2_model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ä
!model_2/model_1/dense_1/BiasAdd_1BiasAdd*model_2/model_1/dense_1/MatMul_1:product:08model_2/model_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¡
model_2/lambda/subSub(model_2/model_1/dense_1/BiasAdd:output:0*model_2/model_1/dense_1/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
model_2/lambda/SquareSquaremodel_2/lambda/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@f
$model_2/lambda/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¦
model_2/lambda/SumSummodel_2/lambda/Square:y:0-model_2/lambda/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(]
model_2/lambda/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
model_2/lambda/MaximumMaximummodel_2/lambda/Sum:output:0!model_2/lambda/Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
model_2/lambda/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model_2/lambda/Maximum_1Maximummodel_2/lambda/Maximum:z:0model_2/lambda/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
model_2/lambda/SqrtSqrtmodel_2/lambda/Maximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%model_2/dense_2/MatMul/ReadVariableOpReadVariableOp.model_2_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
model_2/dense_2/MatMulMatMulmodel_2/lambda/Sqrt:y:0-model_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model_2/dense_2/BiasAddBiasAdd model_2/dense_2/MatMul:product:0.model_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
model_2/dense_2/SigmoidSigmoid model_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitymodel_2/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿý 
NoOpNoOp'^model_2/dense_2/BiasAdd/ReadVariableOp&^model_2/dense_2/MatMul/ReadVariableOp/^model_2/model_1/dense_1/BiasAdd/ReadVariableOp1^model_2/model_1/dense_1/BiasAdd_1/ReadVariableOp.^model_2/model_1/dense_1/MatMul/ReadVariableOp0^model_2/model_1/dense_1/MatMul_1/ReadVariableOp:^model_2/model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp:^model_2/model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp<^model_2/model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp9^model_2/model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp;^model_2/model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&model_2/dense_2/BiasAdd/ReadVariableOp&model_2/dense_2/BiasAdd/ReadVariableOp2N
%model_2/dense_2/MatMul/ReadVariableOp%model_2/dense_2/MatMul/ReadVariableOp2`
.model_2/model_1/dense_1/BiasAdd/ReadVariableOp.model_2/model_1/dense_1/BiasAdd/ReadVariableOp2d
0model_2/model_1/dense_1/BiasAdd_1/ReadVariableOp0model_2/model_1/dense_1/BiasAdd_1/ReadVariableOp2^
-model_2/model_1/dense_1/MatMul/ReadVariableOp-model_2/model_1/dense_1/MatMul/ReadVariableOp2b
/model_2/model_1/dense_1/MatMul_1/ReadVariableOp/model_2/model_1/dense_1/MatMul_1/ReadVariableOp2v
9model_2/model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp2v
9model_2/model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp9model_2/model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp2z
;model_2/model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp;model_2/model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp2t
8model_2/model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp8model_2/model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp2x
:model_2/model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:model_2/model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_3:ZV
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
!
_user_specified_name	input_4
Àª
3
D__inference_model_2_layer_call_and_return_conditional_losses_1037623
inputs_0
inputs_1S
9model_1_vgg19_block1_conv1_conv2d_readvariableop_resource:@H
:model_1_vgg19_block1_conv1_biasadd_readvariableop_resource:@S
9model_1_vgg19_block1_conv2_conv2d_readvariableop_resource:@@H
:model_1_vgg19_block1_conv2_biasadd_readvariableop_resource:@T
9model_1_vgg19_block2_conv1_conv2d_readvariableop_resource:@I
:model_1_vgg19_block2_conv1_biasadd_readvariableop_resource:	U
9model_1_vgg19_block2_conv2_conv2d_readvariableop_resource:I
:model_1_vgg19_block2_conv2_biasadd_readvariableop_resource:	U
9model_1_vgg19_block3_conv1_conv2d_readvariableop_resource:I
:model_1_vgg19_block3_conv1_biasadd_readvariableop_resource:	U
9model_1_vgg19_block3_conv2_conv2d_readvariableop_resource:I
:model_1_vgg19_block3_conv2_biasadd_readvariableop_resource:	U
9model_1_vgg19_block3_conv3_conv2d_readvariableop_resource:I
:model_1_vgg19_block3_conv3_biasadd_readvariableop_resource:	U
9model_1_vgg19_block3_conv4_conv2d_readvariableop_resource:I
:model_1_vgg19_block3_conv4_biasadd_readvariableop_resource:	U
9model_1_vgg19_block4_conv1_conv2d_readvariableop_resource:I
:model_1_vgg19_block4_conv1_biasadd_readvariableop_resource:	U
9model_1_vgg19_block4_conv2_conv2d_readvariableop_resource:I
:model_1_vgg19_block4_conv2_biasadd_readvariableop_resource:	U
9model_1_vgg19_block4_conv3_conv2d_readvariableop_resource:I
:model_1_vgg19_block4_conv3_biasadd_readvariableop_resource:	U
9model_1_vgg19_block4_conv4_conv2d_readvariableop_resource:I
:model_1_vgg19_block4_conv4_biasadd_readvariableop_resource:	U
9model_1_vgg19_block5_conv1_conv2d_readvariableop_resource:I
:model_1_vgg19_block5_conv1_biasadd_readvariableop_resource:	U
9model_1_vgg19_block5_conv2_conv2d_readvariableop_resource:I
:model_1_vgg19_block5_conv2_biasadd_readvariableop_resource:	U
9model_1_vgg19_block5_conv3_conv2d_readvariableop_resource:I
:model_1_vgg19_block5_conv3_biasadd_readvariableop_resource:	U
9model_1_vgg19_block5_conv4_conv2d_readvariableop_resource:I
:model_1_vgg19_block5_conv4_biasadd_readvariableop_resource:	A
.model_1_dense_1_matmul_readvariableop_resource:	@=
/model_1_dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢&model_1/dense_1/BiasAdd/ReadVariableOp¢(model_1/dense_1/BiasAdd_1/ReadVariableOp¢%model_1/dense_1/MatMul/ReadVariableOp¢'model_1/dense_1/MatMul_1/ReadVariableOp¢1model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp¢3model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp¢2model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp¢3model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp¢2model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp¢3model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp¢2model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp¢3model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp¢2model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp¢3model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp¢2model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp¢3model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp¢2model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp¢3model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp¢2model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp¢3model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp¢2model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp¢3model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp¢2model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp¢3model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp¢2model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp¢3model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp¢2model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp¢3model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp¢2model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp¢3model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp¢2model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp¢3model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp¢2model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp¢3model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp¢2model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp¢1model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp¢3model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp¢0model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp¢2model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp²
0model_1/vgg19/block1_conv1/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ó
!model_1/vgg19/block1_conv1/Conv2DConv2Dinputs_08model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¨
1model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ð
"model_1/vgg19/block1_conv1/BiasAddBiasAdd*model_1/vgg19/block1_conv1/Conv2D:output:09model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
model_1/vgg19/block1_conv1/ReluRelu+model_1/vgg19/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@²
0model_1/vgg19/block1_conv2/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ø
!model_1/vgg19/block1_conv2/Conv2DConv2D-model_1/vgg19/block1_conv1/Relu:activations:08model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
¨
1model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ð
"model_1/vgg19/block1_conv2/BiasAddBiasAdd*model_1/vgg19/block1_conv2/Conv2D:output:09model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
model_1/vgg19/block1_conv2/ReluRelu+model_1/vgg19/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@È
!model_1/vgg19/block1_pool/MaxPoolMaxPool-model_1/vgg19/block1_conv2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
³
0model_1/vgg19/block2_conv1/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ô
!model_1/vgg19/block2_conv1/Conv2DConv2D*model_1/vgg19/block1_pool/MaxPool:output:08model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
©
1model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block2_conv1/BiasAddBiasAdd*model_1/vgg19/block2_conv1/Conv2D:output:09model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
model_1/vgg19/block2_conv1/ReluRelu+model_1/vgg19/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd´
0model_1/vgg19/block2_conv2/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block2_conv2/Conv2DConv2D-model_1/vgg19/block2_conv1/Relu:activations:08model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
©
1model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block2_conv2/BiasAddBiasAdd*model_1/vgg19/block2_conv2/Conv2D:output:09model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
model_1/vgg19/block2_conv2/ReluRelu+model_1/vgg19/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÉ
!model_1/vgg19/block2_pool/MaxPoolMaxPool-model_1/vgg19/block2_conv2/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
´
0model_1/vgg19/block3_conv1/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ô
!model_1/vgg19/block3_conv1/Conv2DConv2D*model_1/vgg19/block2_pool/MaxPool:output:08model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
©
1model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block3_conv1/BiasAddBiasAdd*model_1/vgg19/block3_conv1/Conv2D:output:09model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
model_1/vgg19/block3_conv1/ReluRelu+model_1/vgg19/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22´
0model_1/vgg19/block3_conv2/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block3_conv2/Conv2DConv2D-model_1/vgg19/block3_conv1/Relu:activations:08model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
©
1model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block3_conv2/BiasAddBiasAdd*model_1/vgg19/block3_conv2/Conv2D:output:09model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
model_1/vgg19/block3_conv2/ReluRelu+model_1/vgg19/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22´
0model_1/vgg19/block3_conv3/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block3_conv3/Conv2DConv2D-model_1/vgg19/block3_conv2/Relu:activations:08model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
©
1model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block3_conv3/BiasAddBiasAdd*model_1/vgg19/block3_conv3/Conv2D:output:09model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
model_1/vgg19/block3_conv3/ReluRelu+model_1/vgg19/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22´
0model_1/vgg19/block3_conv4/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block3_conv4/Conv2DConv2D-model_1/vgg19/block3_conv3/Relu:activations:08model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
©
1model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block3_conv4/BiasAddBiasAdd*model_1/vgg19/block3_conv4/Conv2D:output:09model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
model_1/vgg19/block3_conv4/ReluRelu+model_1/vgg19/block3_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22É
!model_1/vgg19/block3_pool/MaxPoolMaxPool-model_1/vgg19/block3_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
´
0model_1/vgg19/block4_conv1/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ô
!model_1/vgg19/block4_conv1/Conv2DConv2D*model_1/vgg19/block3_pool/MaxPool:output:08model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block4_conv1/BiasAddBiasAdd*model_1/vgg19/block4_conv1/Conv2D:output:09model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block4_conv1/ReluRelu+model_1/vgg19/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block4_conv2/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block4_conv2/Conv2DConv2D-model_1/vgg19/block4_conv1/Relu:activations:08model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block4_conv2/BiasAddBiasAdd*model_1/vgg19/block4_conv2/Conv2D:output:09model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block4_conv2/ReluRelu+model_1/vgg19/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block4_conv3/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block4_conv3/Conv2DConv2D-model_1/vgg19/block4_conv2/Relu:activations:08model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block4_conv3/BiasAddBiasAdd*model_1/vgg19/block4_conv3/Conv2D:output:09model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block4_conv3/ReluRelu+model_1/vgg19/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block4_conv4/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block4_conv4/Conv2DConv2D-model_1/vgg19/block4_conv3/Relu:activations:08model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block4_conv4/BiasAddBiasAdd*model_1/vgg19/block4_conv4/Conv2D:output:09model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block4_conv4/ReluRelu+model_1/vgg19/block4_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
!model_1/vgg19/block4_pool/MaxPoolMaxPool-model_1/vgg19/block4_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
´
0model_1/vgg19/block5_conv1/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ô
!model_1/vgg19/block5_conv1/Conv2DConv2D*model_1/vgg19/block4_pool/MaxPool:output:08model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block5_conv1/BiasAddBiasAdd*model_1/vgg19/block5_conv1/Conv2D:output:09model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block5_conv1/ReluRelu+model_1/vgg19/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block5_conv2/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block5_conv2/Conv2DConv2D-model_1/vgg19/block5_conv1/Relu:activations:08model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block5_conv2/BiasAddBiasAdd*model_1/vgg19/block5_conv2/Conv2D:output:09model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block5_conv2/ReluRelu+model_1/vgg19/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block5_conv3/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block5_conv3/Conv2DConv2D-model_1/vgg19/block5_conv2/Relu:activations:08model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block5_conv3/BiasAddBiasAdd*model_1/vgg19/block5_conv3/Conv2D:output:09model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block5_conv3/ReluRelu+model_1/vgg19/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
0model_1/vgg19/block5_conv4/Conv2D/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0÷
!model_1/vgg19/block5_conv4/Conv2DConv2D-model_1/vgg19/block5_conv3/Relu:activations:08model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
©
1model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ï
"model_1/vgg19/block5_conv4/BiasAddBiasAdd*model_1/vgg19/block5_conv4/Conv2D:output:09model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/vgg19/block5_conv4/ReluRelu+model_1/vgg19/block5_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
!model_1/vgg19/block5_pool/MaxPoolMaxPool-model_1/vgg19/block5_conv4/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

9model_1/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ò
'model_1/global_average_pooling2d_1/MeanMean*model_1/vgg19/block5_pool/MaxPool:output:0Bmodel_1/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0³
model_1/dense_1/MatMulMatMul0model_1/global_average_pooling2d_1/Mean:output:0-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¦
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@´
2model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0×
#model_1/vgg19/block1_conv1/Conv2D_1Conv2Dinputs_1:model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
ª
3model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
$model_1/vgg19/block1_conv1/BiasAdd_1BiasAdd,model_1/vgg19/block1_conv1/Conv2D_1:output:0;model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
!model_1/vgg19/block1_conv1/Relu_1Relu-model_1/vgg19/block1_conv1/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@´
2model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0þ
#model_1/vgg19/block1_conv2/Conv2D_1Conv2D/model_1/vgg19/block1_conv1/Relu_1:activations:0:model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@*
paddingSAME*
strides
ª
3model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
$model_1/vgg19/block1_conv2/BiasAdd_1BiasAdd,model_1/vgg19/block1_conv2/Conv2D_1:output:0;model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@
!model_1/vgg19/block1_conv2/Relu_1Relu-model_1/vgg19/block1_conv2/BiasAdd_1:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ@Ì
#model_1/vgg19/block1_pool/MaxPool_1MaxPool/model_1/vgg19/block1_conv2/Relu_1:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd@*
ksize
*
paddingVALID*
strides
µ
2model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ú
#model_1/vgg19/block2_conv1/Conv2D_1Conv2D,model_1/vgg19/block1_pool/MaxPool_1:output:0:model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
«
3model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block2_conv1/BiasAdd_1BiasAdd,model_1/vgg19/block2_conv1/Conv2D_1:output:0;model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!model_1/vgg19/block2_conv1/Relu_1Relu-model_1/vgg19/block2_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¶
2model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block2_conv2/Conv2D_1Conv2D/model_1/vgg19/block2_conv1/Relu_1:activations:0:model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingSAME*
strides
«
3model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block2_conv2/BiasAdd_1BiasAdd,model_1/vgg19/block2_conv2/Conv2D_1:output:0;model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!model_1/vgg19/block2_conv2/Relu_1Relu-model_1/vgg19/block2_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÍ
#model_1/vgg19/block2_pool/MaxPool_1MaxPool/model_1/vgg19/block2_conv2/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
ksize
*
paddingVALID*
strides
¶
2model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
#model_1/vgg19/block3_conv1/Conv2D_1Conv2D,model_1/vgg19/block2_pool/MaxPool_1:output:0:model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
3model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block3_conv1/BiasAdd_1BiasAdd,model_1/vgg19/block3_conv1/Conv2D_1:output:0;model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!model_1/vgg19/block3_conv1/Relu_1Relu-model_1/vgg19/block3_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
2model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block3_conv2/Conv2D_1Conv2D/model_1/vgg19/block3_conv1/Relu_1:activations:0:model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
3model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block3_conv2/BiasAdd_1BiasAdd,model_1/vgg19/block3_conv2/Conv2D_1:output:0;model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!model_1/vgg19/block3_conv2/Relu_1Relu-model_1/vgg19/block3_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
2model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block3_conv3/Conv2D_1Conv2D/model_1/vgg19/block3_conv2/Relu_1:activations:0:model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
3model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block3_conv3/BiasAdd_1BiasAdd,model_1/vgg19/block3_conv3/Conv2D_1:output:0;model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!model_1/vgg19/block3_conv3/Relu_1Relu-model_1/vgg19/block3_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¶
2model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block3_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block3_conv4/Conv2D_1Conv2D/model_1/vgg19/block3_conv3/Relu_1:activations:0:model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
«
3model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block3_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block3_conv4/BiasAdd_1BiasAdd,model_1/vgg19/block3_conv4/Conv2D_1:output:0;model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!model_1/vgg19/block3_conv4/Relu_1Relu-model_1/vgg19/block3_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Í
#model_1/vgg19/block3_pool/MaxPool_1MaxPool/model_1/vgg19/block3_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¶
2model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
#model_1/vgg19/block4_conv1/Conv2D_1Conv2D,model_1/vgg19/block3_pool/MaxPool_1:output:0:model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block4_conv1/BiasAdd_1BiasAdd,model_1/vgg19/block4_conv1/Conv2D_1:output:0;model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block4_conv1/Relu_1Relu-model_1/vgg19/block4_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block4_conv2/Conv2D_1Conv2D/model_1/vgg19/block4_conv1/Relu_1:activations:0:model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block4_conv2/BiasAdd_1BiasAdd,model_1/vgg19/block4_conv2/Conv2D_1:output:0;model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block4_conv2/Relu_1Relu-model_1/vgg19/block4_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block4_conv3/Conv2D_1Conv2D/model_1/vgg19/block4_conv2/Relu_1:activations:0:model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block4_conv3/BiasAdd_1BiasAdd,model_1/vgg19/block4_conv3/Conv2D_1:output:0;model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block4_conv3/Relu_1Relu-model_1/vgg19/block4_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block4_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block4_conv4/Conv2D_1Conv2D/model_1/vgg19/block4_conv3/Relu_1:activations:0:model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block4_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block4_conv4/BiasAdd_1BiasAdd,model_1/vgg19/block4_conv4/Conv2D_1:output:0;model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block4_conv4/Relu_1Relu-model_1/vgg19/block4_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
#model_1/vgg19/block4_pool/MaxPool_1MaxPool/model_1/vgg19/block4_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¶
2model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ú
#model_1/vgg19/block5_conv1/Conv2D_1Conv2D,model_1/vgg19/block4_pool/MaxPool_1:output:0:model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block5_conv1/BiasAdd_1BiasAdd,model_1/vgg19/block5_conv1/Conv2D_1:output:0;model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block5_conv1/Relu_1Relu-model_1/vgg19/block5_conv1/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block5_conv2/Conv2D_1Conv2D/model_1/vgg19/block5_conv1/Relu_1:activations:0:model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block5_conv2/BiasAdd_1BiasAdd,model_1/vgg19/block5_conv2/Conv2D_1:output:0;model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block5_conv2/Relu_1Relu-model_1/vgg19/block5_conv2/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block5_conv3/Conv2D_1Conv2D/model_1/vgg19/block5_conv2/Relu_1:activations:0:model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block5_conv3/BiasAdd_1BiasAdd,model_1/vgg19/block5_conv3/Conv2D_1:output:0;model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block5_conv3/Relu_1Relu-model_1/vgg19/block5_conv3/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
2model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOpReadVariableOp9model_1_vgg19_block5_conv4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ý
#model_1/vgg19/block5_conv4/Conv2D_1Conv2D/model_1/vgg19/block5_conv3/Relu_1:activations:0:model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
«
3model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOpReadVariableOp:model_1_vgg19_block5_conv4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
$model_1/vgg19/block5_conv4/BiasAdd_1BiasAdd,model_1/vgg19/block5_conv4/Conv2D_1:output:0;model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model_1/vgg19/block5_conv4/Relu_1Relu-model_1/vgg19/block5_conv4/BiasAdd_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
#model_1/vgg19/block5_pool/MaxPool_1MaxPool/model_1/vgg19/block5_conv4/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

;model_1/global_average_pooling2d_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ø
)model_1/global_average_pooling2d_1/Mean_1Mean,model_1/vgg19/block5_pool/MaxPool_1:output:0Dmodel_1/global_average_pooling2d_1/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0¹
model_1/dense_1/MatMul_1MatMul2model_1/global_average_pooling2d_1/Mean_1:output:0/model_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(model_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¬
model_1/dense_1/BiasAdd_1BiasAdd"model_1/dense_1/MatMul_1:product:00model_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

lambda/subSub model_1/dense_1/BiasAdd:output:0"model_1/dense_1/BiasAdd_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
lambda/SquareSquarelambda/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@^
lambda/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :

lambda/SumSumlambda/Square:y:0%lambda/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
	keep_dims(U
lambda/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3{
lambda/MaximumMaximumlambda/Sum:output:0lambda/Maximum/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
lambda/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
lambda/Maximum_1Maximumlambda/Maximum:z:0lambda/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lambda/SqrtSqrtlambda/Maximum_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_2/MatMulMatMullambda/Sqrt:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp'^model_1/dense_1/BiasAdd/ReadVariableOp)^model_1/dense_1/BiasAdd_1/ReadVariableOp&^model_1/dense_1/MatMul/ReadVariableOp(^model_1/dense_1/MatMul_1/ReadVariableOp2^model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp4^model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp3^model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp2^model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp4^model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp3^model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp2^model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp4^model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp3^model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp2^model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp4^model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp3^model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp2^model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp4^model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp3^model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp2^model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp4^model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp3^model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp2^model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp4^model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp3^model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp2^model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp4^model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp3^model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp2^model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp4^model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp3^model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp2^model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp4^model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp3^model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp2^model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp4^model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp3^model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp2^model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp4^model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp3^model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp2^model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp4^model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp3^model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp2^model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp4^model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp3^model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp2^model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp4^model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp3^model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp2^model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp4^model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp1^model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp3^model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2P
&model_1/dense_1/BiasAdd/ReadVariableOp&model_1/dense_1/BiasAdd/ReadVariableOp2T
(model_1/dense_1/BiasAdd_1/ReadVariableOp(model_1/dense_1/BiasAdd_1/ReadVariableOp2N
%model_1/dense_1/MatMul/ReadVariableOp%model_1/dense_1/MatMul/ReadVariableOp2R
'model_1/dense_1/MatMul_1/ReadVariableOp'model_1/dense_1/MatMul_1/ReadVariableOp2f
1model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp1model_1/vgg19/block1_conv1/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp3model_1/vgg19/block1_conv1/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp0model_1/vgg19/block1_conv1/Conv2D/ReadVariableOp2h
2model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp2model_1/vgg19/block1_conv1/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp1model_1/vgg19/block1_conv2/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp3model_1/vgg19/block1_conv2/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp0model_1/vgg19/block1_conv2/Conv2D/ReadVariableOp2h
2model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp2model_1/vgg19/block1_conv2/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp1model_1/vgg19/block2_conv1/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp3model_1/vgg19/block2_conv1/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp0model_1/vgg19/block2_conv1/Conv2D/ReadVariableOp2h
2model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp2model_1/vgg19/block2_conv1/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp1model_1/vgg19/block2_conv2/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp3model_1/vgg19/block2_conv2/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp0model_1/vgg19/block2_conv2/Conv2D/ReadVariableOp2h
2model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp2model_1/vgg19/block2_conv2/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp1model_1/vgg19/block3_conv1/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp3model_1/vgg19/block3_conv1/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp0model_1/vgg19/block3_conv1/Conv2D/ReadVariableOp2h
2model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp2model_1/vgg19/block3_conv1/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp1model_1/vgg19/block3_conv2/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp3model_1/vgg19/block3_conv2/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp0model_1/vgg19/block3_conv2/Conv2D/ReadVariableOp2h
2model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp2model_1/vgg19/block3_conv2/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp1model_1/vgg19/block3_conv3/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp3model_1/vgg19/block3_conv3/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp0model_1/vgg19/block3_conv3/Conv2D/ReadVariableOp2h
2model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp2model_1/vgg19/block3_conv3/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp1model_1/vgg19/block3_conv4/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp3model_1/vgg19/block3_conv4/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp0model_1/vgg19/block3_conv4/Conv2D/ReadVariableOp2h
2model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp2model_1/vgg19/block3_conv4/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp1model_1/vgg19/block4_conv1/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp3model_1/vgg19/block4_conv1/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp0model_1/vgg19/block4_conv1/Conv2D/ReadVariableOp2h
2model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp2model_1/vgg19/block4_conv1/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp1model_1/vgg19/block4_conv2/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp3model_1/vgg19/block4_conv2/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp0model_1/vgg19/block4_conv2/Conv2D/ReadVariableOp2h
2model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp2model_1/vgg19/block4_conv2/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp1model_1/vgg19/block4_conv3/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp3model_1/vgg19/block4_conv3/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp0model_1/vgg19/block4_conv3/Conv2D/ReadVariableOp2h
2model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp2model_1/vgg19/block4_conv3/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp1model_1/vgg19/block4_conv4/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp3model_1/vgg19/block4_conv4/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp0model_1/vgg19/block4_conv4/Conv2D/ReadVariableOp2h
2model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp2model_1/vgg19/block4_conv4/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp1model_1/vgg19/block5_conv1/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp3model_1/vgg19/block5_conv1/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp0model_1/vgg19/block5_conv1/Conv2D/ReadVariableOp2h
2model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp2model_1/vgg19/block5_conv1/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp1model_1/vgg19/block5_conv2/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp3model_1/vgg19/block5_conv2/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp0model_1/vgg19/block5_conv2/Conv2D/ReadVariableOp2h
2model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp2model_1/vgg19/block5_conv2/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp1model_1/vgg19/block5_conv3/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp3model_1/vgg19/block5_conv3/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp0model_1/vgg19/block5_conv3/Conv2D/ReadVariableOp2h
2model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp2model_1/vgg19/block5_conv3/Conv2D_1/ReadVariableOp2f
1model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp1model_1/vgg19/block5_conv4/BiasAdd/ReadVariableOp2j
3model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp3model_1/vgg19/block5_conv4/BiasAdd_1/ReadVariableOp2d
0model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp0model_1/vgg19/block5_conv4/Conv2D/ReadVariableOp2h
2model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp2model_1/vgg19/block5_conv4/Conv2D_1/ReadVariableOp:[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
"
_user_specified_name
inputs/1

d
H__inference_block2_pool_layer_call_and_return_conditional_losses_1038677

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
ü
'__inference_vgg19_layer_call_fn_1038305

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	&

unknown_11:

unknown_12:	&

unknown_13:

unknown_14:	&

unknown_15:

unknown_16:	&

unknown_17:

unknown_18:	&

unknown_19:

unknown_20:	&

unknown_21:

unknown_22:	&

unknown_23:

unknown_24:	&

unknown_25:

unknown_26:	&

unknown_27:

unknown_28:	&

unknown_29:

unknown_30:	
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_vgg19_layer_call_and_return_conditional_losses_1035207x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿÈÈ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈÈ
 
_user_specified_nameinputs
þ
¦
.__inference_block5_conv3_layer_call_fn_1038906

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_block5_conv3_layer_call_and_return_conditional_losses_1034795x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block5_conv3_layer_call_and_return_conditional_losses_1034795

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_block5_conv1_layer_call_and_return_conditional_losses_1038877

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*û
serving_defaultç
E
input_3:
serving_default_input_3:0ÿÿÿÿÿÿÿÿÿÈÈ
E
input_4:
serving_default_input_4:0ÿÿÿÿÿÿÿÿÿÈÈ;
dense_20
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÑÁ
Ø
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
£
'iter

(beta_1

)beta_2
	*decay
+learning_ratem mLmMmv vLvMv"
	optimizer
¶
,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31
L32
M33
34
 35"
trackable_list_wrapper
<
L0
M1
2
 3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_model_2_layer_call_fn_1036371
)__inference_model_2_layer_call_fn_1037071
)__inference_model_2_layer_call_fn_1037149
)__inference_model_2_layer_call_fn_1036757À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_model_2_layer_call_and_return_conditional_losses_1037386
D__inference_model_2_layer_call_and_return_conditional_losses_1037623
D__inference_model_2_layer_call_and_return_conditional_losses_1036872
D__inference_model_2_layer_call_and_return_conditional_losses_1036987À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÖBÓ
"__inference__wrapped_model_1034475input_3input_4"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Sserving_default"
signature_map
"
_tf_keras_input_layer
÷
Tlayer-0
Ulayer_with_weights-0
Ulayer-1
Vlayer_with_weights-1
Vlayer-2
Wlayer-3
Xlayer_with_weights-2
Xlayer-4
Ylayer_with_weights-3
Ylayer-5
Zlayer-6
[layer_with_weights-4
[layer-7
\layer_with_weights-5
\layer-8
]layer_with_weights-6
]layer-9
^layer_with_weights-7
^layer-10
_layer-11
`layer_with_weights-8
`layer-12
alayer_with_weights-9
alayer-13
blayer_with_weights-10
blayer-14
clayer_with_weights-11
clayer-15
dlayer-16
elayer_with_weights-12
elayer-17
flayer_with_weights-13
flayer-18
glayer_with_weights-14
glayer-19
hlayer_with_weights-15
hlayer-20
ilayer-21
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_network
¥
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Lkernel
Mbias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
¦
,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31
L32
M33"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
®
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_model_1_layer_call_fn_1035695
)__inference_model_1_layer_call_fn_1037776
)__inference_model_1_layer_call_fn_1037849
)__inference_model_1_layer_call_fn_1035999À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_model_1_layer_call_and_return_conditional_losses_1037978
D__inference_model_1_layer_call_and_return_conditional_losses_1038107
D__inference_model_1_layer_call_and_return_conditional_losses_1036074
D__inference_model_1_layer_call_and_return_conditional_losses_1036149À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
(__inference_lambda_layer_call_fn_1038113
(__inference_lambda_layer_call_fn_1038119À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
C__inference_lambda_layer_call_and_return_conditional_losses_1038133
C__inference_lambda_layer_call_and_return_conditional_losses_1038147À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 :2dense_2/kernel
:2dense_2/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_2_layer_call_fn_1038156¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_2_layer_call_and_return_conditional_losses_1038167¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@2block2_conv1/kernel
 :2block2_conv1/bias
/:-2block2_conv2/kernel
 :2block2_conv2/bias
/:-2block3_conv1/kernel
 :2block3_conv1/bias
/:-2block3_conv2/kernel
 :2block3_conv2/bias
/:-2block3_conv3/kernel
 :2block3_conv3/bias
/:-2block3_conv4/kernel
 :2block3_conv4/bias
/:-2block4_conv1/kernel
 :2block4_conv1/bias
/:-2block4_conv2/kernel
 :2block4_conv2/bias
/:-2block4_conv3/kernel
 :2block4_conv3/bias
/:-2block4_conv4/kernel
 :2block4_conv4/bias
/:-2block5_conv1/kernel
 :2block5_conv1/bias
/:-2block5_conv2/kernel
 :2block5_conv2/bias
/:-2block5_conv3/kernel
 :2block5_conv3/bias
/:-2block5_conv4/kernel
 :2block5_conv4/bias
!:	@2dense_1/kernel
:@2dense_1/bias

,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÓBÐ
%__inference_signature_wrapper_1037703input_3input_4"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_tf_keras_input_layer
Á

,kernel
-bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

.kernel
/bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

0kernel
1bias
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

2kernel
3bias
¥	variables
¦trainable_variables
§regularization_losses
¨	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"
_tf_keras_layer
«
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

4kernel
5bias
±	variables
²trainable_variables
³regularization_losses
´	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

6kernel
7bias
·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
»__call__
+¼&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

8kernel
9bias
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

:kernel
;bias
Ã	variables
Ätrainable_variables
Åregularization_losses
Æ	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"
_tf_keras_layer
«
É	variables
Êtrainable_variables
Ëregularization_losses
Ì	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

<kernel
=bias
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

>kernel
?bias
Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

@kernel
Abias
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Bkernel
Cbias
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"
_tf_keras_layer
«
ç	variables
ètrainable_variables
éregularization_losses
ê	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Dkernel
Ebias
í	variables
îtrainable_variables
ïregularization_losses
ð	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Fkernel
Gbias
ó	variables
ôtrainable_variables
õregularization_losses
ö	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Hkernel
Ibias
ù	variables
útrainable_variables
ûregularization_losses
ü	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Jkernel
Kbias
ÿ	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
ê2ç
'__inference_vgg19_layer_call_fn_1034887
'__inference_vgg19_layer_call_fn_1038236
'__inference_vgg19_layer_call_fn_1038305
'__inference_vgg19_layer_call_fn_1035343À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
B__inference_vgg19_layer_call_and_return_conditional_losses_1038426
B__inference_vgg19_layer_call_and_return_conditional_losses_1038547
B__inference_vgg19_layer_call_and_return_conditional_losses_1035432
B__inference_vgg19_layer_call_and_return_conditional_losses_1035521À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
æ2ã
<__inference_global_average_pooling2d_1_layer_call_fn_1038552¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2þ
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_1038558¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_1_layer_call_fn_1038567¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_1_layer_call_and_return_conditional_losses_1038577¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count
 
_fn_kwargs
¡	variables
¢	keras_api"
_tf_keras_metric
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block1_conv1_layer_call_fn_1038586¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block1_conv1_layer_call_and_return_conditional_losses_1038597¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block1_conv2_layer_call_fn_1038606¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block1_conv2_layer_call_and_return_conditional_losses_1038617¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block1_pool_layer_call_fn_1038622¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block1_pool_layer_call_and_return_conditional_losses_1038627¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
	variables
 trainable_variables
¡regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block2_conv1_layer_call_fn_1038636¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block2_conv1_layer_call_and_return_conditional_losses_1038647¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
¥	variables
¦trainable_variables
§regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block2_conv2_layer_call_fn_1038656¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block2_conv2_layer_call_and_return_conditional_losses_1038667¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
«	variables
¬trainable_variables
­regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block2_pool_layer_call_fn_1038672¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block2_pool_layer_call_and_return_conditional_losses_1038677¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
±	variables
²trainable_variables
³regularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block3_conv1_layer_call_fn_1038686¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block3_conv1_layer_call_and_return_conditional_losses_1038697¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
·	variables
¸trainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block3_conv2_layer_call_fn_1038706¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block3_conv2_layer_call_and_return_conditional_losses_1038717¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block3_conv3_layer_call_fn_1038726¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block3_conv3_layer_call_and_return_conditional_losses_1038737¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
Ã	variables
Ätrainable_variables
Åregularization_losses
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block3_conv4_layer_call_fn_1038746¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block3_conv4_layer_call_and_return_conditional_losses_1038757¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
É	variables
Êtrainable_variables
Ëregularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block3_pool_layer_call_fn_1038762¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block3_pool_layer_call_and_return_conditional_losses_1038767¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block4_conv1_layer_call_fn_1038776¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block4_conv1_layer_call_and_return_conditional_losses_1038787¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block4_conv2_layer_call_fn_1038796¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block4_conv2_layer_call_and_return_conditional_losses_1038807¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block4_conv3_layer_call_fn_1038816¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block4_conv3_layer_call_and_return_conditional_losses_1038827¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
á	variables
âtrainable_variables
ãregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block4_conv4_layer_call_fn_1038836¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block4_conv4_layer_call_and_return_conditional_losses_1038847¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
ç	variables
ètrainable_variables
éregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block4_pool_layer_call_fn_1038852¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block4_pool_layer_call_and_return_conditional_losses_1038857¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
í	variables
îtrainable_variables
ïregularization_losses
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block5_conv1_layer_call_fn_1038866¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block5_conv1_layer_call_and_return_conditional_losses_1038877¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
ó	variables
ôtrainable_variables
õregularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block5_conv2_layer_call_fn_1038886¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block5_conv2_layer_call_and_return_conditional_losses_1038897¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
ù	variables
útrainable_variables
ûregularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block5_conv3_layer_call_fn_1038906¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block5_conv3_layer_call_and_return_conditional_losses_1038917¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ÿ	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_block5_conv4_layer_call_fn_1038926¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_block5_conv4_layer_call_and_return_conditional_losses_1038937¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
×2Ô
-__inference_block5_pool_layer_call_fn_1038942¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_block5_pool_layer_call_and_return_conditional_losses_1038947¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
?19
@20
A21
B22
C23
D24
E25
F26
G27
H28
I29
J30
K31"
trackable_list_wrapper
Æ
T0
U1
V2
W3
X4
Y5
Z6
[7
\8
]9
^10
_11
`12
a13
b14
c15
d16
e17
f18
g19
h20
i21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
¡	variables"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
&:$	@2Adam/dense_1/kernel/m
:@2Adam/dense_1/bias/m
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
&:$	@2Adam/dense_1/kernel/v
:@2Adam/dense_1/bias/vî
"__inference__wrapped_model_1034475Ç$,-./0123456789:;<=>?@ABCDEFGHIJKLM l¢i
b¢_
]Z
+(
input_3ÿÿÿÿÿÿÿÿÿÈÈ
+(
input_4ÿÿÿÿÿÿÿÿÿÈÈ
ª "1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿ½
I__inference_block1_conv1_layer_call_and_return_conditional_losses_1038597p,-9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿÈÈ@
 
.__inference_block1_conv1_layer_call_fn_1038586c,-9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
ª ""ÿÿÿÿÿÿÿÿÿÈÈ@½
I__inference_block1_conv2_layer_call_and_return_conditional_losses_1038617p./9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿÈÈ@
 
.__inference_block1_conv2_layer_call_fn_1038606c./9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ@
ª ""ÿÿÿÿÿÿÿÿÿÈÈ@ë
H__inference_block1_pool_layer_call_and_return_conditional_losses_1038627R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_block1_pool_layer_call_fn_1038622R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
I__inference_block2_conv1_layer_call_and_return_conditional_losses_1038647m017¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿdd
 
.__inference_block2_conv1_layer_call_fn_1038636`017¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd@
ª "!ÿÿÿÿÿÿÿÿÿdd»
I__inference_block2_conv2_layer_call_and_return_conditional_losses_1038667n238¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿdd
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿdd
 
.__inference_block2_conv2_layer_call_fn_1038656a238¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿdd
ª "!ÿÿÿÿÿÿÿÿÿddë
H__inference_block2_pool_layer_call_and_return_conditional_losses_1038677R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_block2_pool_layer_call_fn_1038672R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
I__inference_block3_conv1_layer_call_and_return_conditional_losses_1038697n458¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ22
 
.__inference_block3_conv1_layer_call_fn_1038686a458¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª "!ÿÿÿÿÿÿÿÿÿ22»
I__inference_block3_conv2_layer_call_and_return_conditional_losses_1038717n678¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ22
 
.__inference_block3_conv2_layer_call_fn_1038706a678¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª "!ÿÿÿÿÿÿÿÿÿ22»
I__inference_block3_conv3_layer_call_and_return_conditional_losses_1038737n898¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ22
 
.__inference_block3_conv3_layer_call_fn_1038726a898¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª "!ÿÿÿÿÿÿÿÿÿ22»
I__inference_block3_conv4_layer_call_and_return_conditional_losses_1038757n:;8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ22
 
.__inference_block3_conv4_layer_call_fn_1038746a:;8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ22
ª "!ÿÿÿÿÿÿÿÿÿ22ë
H__inference_block3_pool_layer_call_and_return_conditional_losses_1038767R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_block3_pool_layer_call_fn_1038762R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
I__inference_block4_conv1_layer_call_and_return_conditional_losses_1038787n<=8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_block4_conv1_layer_call_fn_1038776a<=8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ»
I__inference_block4_conv2_layer_call_and_return_conditional_losses_1038807n>?8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_block4_conv2_layer_call_fn_1038796a>?8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ»
I__inference_block4_conv3_layer_call_and_return_conditional_losses_1038827n@A8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_block4_conv3_layer_call_fn_1038816a@A8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ»
I__inference_block4_conv4_layer_call_and_return_conditional_losses_1038847nBC8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_block4_conv4_layer_call_fn_1038836aBC8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿë
H__inference_block4_pool_layer_call_and_return_conditional_losses_1038857R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_block4_pool_layer_call_fn_1038852R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
I__inference_block5_conv1_layer_call_and_return_conditional_losses_1038877nDE8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_block5_conv1_layer_call_fn_1038866aDE8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ»
I__inference_block5_conv2_layer_call_and_return_conditional_losses_1038897nFG8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_block5_conv2_layer_call_fn_1038886aFG8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ»
I__inference_block5_conv3_layer_call_and_return_conditional_losses_1038917nHI8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_block5_conv3_layer_call_fn_1038906aHI8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ»
I__inference_block5_conv4_layer_call_and_return_conditional_losses_1038937nJK8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_block5_conv4_layer_call_fn_1038926aJK8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿë
H__inference_block5_pool_layer_call_and_return_conditional_losses_1038947R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_block5_pool_layer_call_fn_1038942R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
D__inference_dense_1_layer_call_and_return_conditional_losses_1038577]LM0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
)__inference_dense_1_layer_call_fn_1038567PLM0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_2_layer_call_and_return_conditional_losses_1038167\ /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_2_layer_call_fn_1038156O /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿà
W__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_1038558R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ·
<__inference_global_average_pooling2d_1_layer_call_fn_1038552wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÓ
C__inference_lambda_layer_call_and_return_conditional_losses_1038133b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@

 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ó
C__inference_lambda_layer_call_and_return_conditional_losses_1038147b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@

 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ª
(__inference_lambda_layer_call_fn_1038113~b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@

 
p 
ª "ÿÿÿÿÿÿÿÿÿª
(__inference_lambda_layer_call_fn_1038119~b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ@
"
inputs/1ÿÿÿÿÿÿÿÿÿ@

 
p
ª "ÿÿÿÿÿÿÿÿÿØ
D__inference_model_1_layer_call_and_return_conditional_losses_1036074",-./0123456789:;<=>?@ABCDEFGHIJKLMB¢?
8¢5
+(
input_5ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 Ø
D__inference_model_1_layer_call_and_return_conditional_losses_1036149",-./0123456789:;<=>?@ABCDEFGHIJKLMB¢?
8¢5
+(
input_5ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ×
D__inference_model_1_layer_call_and_return_conditional_losses_1037978",-./0123456789:;<=>?@ABCDEFGHIJKLMA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ×
D__inference_model_1_layer_call_and_return_conditional_losses_1038107",-./0123456789:;<=>?@ABCDEFGHIJKLMA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 °
)__inference_model_1_layer_call_fn_1035695",-./0123456789:;<=>?@ABCDEFGHIJKLMB¢?
8¢5
+(
input_5ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@°
)__inference_model_1_layer_call_fn_1035999",-./0123456789:;<=>?@ABCDEFGHIJKLMB¢?
8¢5
+(
input_5ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿ@¯
)__inference_model_1_layer_call_fn_1037776",-./0123456789:;<=>?@ABCDEFGHIJKLMA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ@¯
)__inference_model_1_layer_call_fn_1037849",-./0123456789:;<=>?@ABCDEFGHIJKLMA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿ@
D__inference_model_2_layer_call_and_return_conditional_losses_1036872Ã$,-./0123456789:;<=>?@ABCDEFGHIJKLM t¢q
j¢g
]Z
+(
input_3ÿÿÿÿÿÿÿÿÿÈÈ
+(
input_4ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
D__inference_model_2_layer_call_and_return_conditional_losses_1036987Ã$,-./0123456789:;<=>?@ABCDEFGHIJKLM t¢q
j¢g
]Z
+(
input_3ÿÿÿÿÿÿÿÿÿÈÈ
+(
input_4ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
D__inference_model_2_layer_call_and_return_conditional_losses_1037386Å$,-./0123456789:;<=>?@ABCDEFGHIJKLM v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿÈÈ
,)
inputs/1ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
D__inference_model_2_layer_call_and_return_conditional_losses_1037623Å$,-./0123456789:;<=>?@ABCDEFGHIJKLM v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿÈÈ
,)
inputs/1ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ä
)__inference_model_2_layer_call_fn_1036371¶$,-./0123456789:;<=>?@ABCDEFGHIJKLM t¢q
j¢g
]Z
+(
input_3ÿÿÿÿÿÿÿÿÿÈÈ
+(
input_4ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿä
)__inference_model_2_layer_call_fn_1036757¶$,-./0123456789:;<=>?@ABCDEFGHIJKLM t¢q
j¢g
]Z
+(
input_3ÿÿÿÿÿÿÿÿÿÈÈ
+(
input_4ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿæ
)__inference_model_2_layer_call_fn_1037071¸$,-./0123456789:;<=>?@ABCDEFGHIJKLM v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿÈÈ
,)
inputs/1ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "ÿÿÿÿÿÿÿÿÿæ
)__inference_model_2_layer_call_fn_1037149¸$,-./0123456789:;<=>?@ABCDEFGHIJKLM v¢s
l¢i
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿÈÈ
,)
inputs/1ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
%__inference_signature_wrapper_1037703Ø$,-./0123456789:;<=>?@ABCDEFGHIJKLM }¢z
¢ 
sªp
6
input_3+(
input_3ÿÿÿÿÿÿÿÿÿÈÈ
6
input_4+(
input_4ÿÿÿÿÿÿÿÿÿÈÈ"1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿÝ
B__inference_vgg19_layer_call_and_return_conditional_losses_1035432 ,-./0123456789:;<=>?@ABCDEFGHIJKB¢?
8¢5
+(
input_6ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ý
B__inference_vgg19_layer_call_and_return_conditional_losses_1035521 ,-./0123456789:;<=>?@ABCDEFGHIJKB¢?
8¢5
+(
input_6ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ü
B__inference_vgg19_layer_call_and_return_conditional_losses_1038426 ,-./0123456789:;<=>?@ABCDEFGHIJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ü
B__inference_vgg19_layer_call_and_return_conditional_losses_1038547 ,-./0123456789:;<=>?@ABCDEFGHIJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 µ
'__inference_vgg19_layer_call_fn_1034887 ,-./0123456789:;<=>?@ABCDEFGHIJKB¢?
8¢5
+(
input_6ÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "!ÿÿÿÿÿÿÿÿÿµ
'__inference_vgg19_layer_call_fn_1035343 ,-./0123456789:;<=>?@ABCDEFGHIJKB¢?
8¢5
+(
input_6ÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "!ÿÿÿÿÿÿÿÿÿ´
'__inference_vgg19_layer_call_fn_1038236 ,-./0123456789:;<=>?@ABCDEFGHIJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ´
'__inference_vgg19_layer_call_fn_1038305 ,-./0123456789:;<=>?@ABCDEFGHIJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÈÈ
p

 
ª "!ÿÿÿÿÿÿÿÿÿ