
вЈ
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
О
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8ыч
|
output_-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*!
shared_nameoutput_-1/kernel
u
$output_-1/kernel/Read/ReadVariableOpReadVariableOpoutput_-1/kernel*
_output_shapes

:2*
dtype0
t
output_-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_nameoutput_-1/bias
m
"output_-1/bias/Read/ReadVariableOpReadVariableOpoutput_-1/bias*
_output_shapes
:2*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:2*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
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
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

Adam/output_-1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/output_-1/kernel/m

+Adam/output_-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_-1/kernel/m*
_output_shapes

:2*
dtype0

Adam/output_-1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/output_-1/bias/m
{
)Adam/output_-1/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_-1/bias/m*
_output_shapes
:2*
dtype0

Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:2*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0

Adam/output_-1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/output_-1/kernel/v

+Adam/output_-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_-1/kernel/v*
_output_shapes

:2*
dtype0

Adam/output_-1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_nameAdam/output_-1/bias/v
{
)Adam/output_-1/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_-1/bias/v*
_output_shapes
:2*
dtype0

Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:2*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ю#
valueФ#BС# BК#

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
_model_paras
comp_args_kw
	optimizer
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
 
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api

"grid_dropout

#1

$iter

%beta_1

&beta_2
	'decay
(learning_ratemTmUmVmWvXvYvZv[

0
1
2
3
 

0
1
2
3
­
)layer_regularization_losses
*non_trainable_variables
+metrics
	trainable_variables
,layer_metrics

regularization_losses

-layers
	variables
 
 
 
 
­
.layer_regularization_losses
/non_trainable_variables
0metrics
trainable_variables
1layer_metrics
regularization_losses

2layers
	variables
\Z
VARIABLE_VALUEoutput_-1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEoutput_-1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
3layer_regularization_losses
4non_trainable_variables
5metrics
trainable_variables
6layer_metrics
regularization_losses

7layers
	variables
 
 
 
­
8layer_regularization_losses
9non_trainable_variables
:metrics
trainable_variables
;layer_metrics
regularization_losses

<layers
	variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
=layer_regularization_losses
>non_trainable_variables
?metrics
trainable_variables
@layer_metrics
regularization_losses

Alayers
 	variables
 

	optimizer
Bmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

C0
D1
E2
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ftotal
	Gcount
H	variables
I	keras_api
D
	Jtotal
	Kcount
L
_fn_kwargs
M	variables
N	keras_api
D
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

H	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

M	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

O0
P1

R	variables
}
VARIABLE_VALUEAdam/output_-1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/output_-1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/output_-1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/output_-1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_inputPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
њ
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputoutput_-1/kerneloutput_-1/biasoutput/kerneloutput/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_3109910
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ы
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$output_-1/kernel/Read/ReadVariableOp"output_-1/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/output_-1/kernel/m/Read/ReadVariableOp)Adam/output_-1/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp+Adam/output_-1/kernel/v/Read/ReadVariableOp)Adam/output_-1/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*$
Tin
2	*
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
 __inference__traced_save_3110167

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameoutput_-1/kerneloutput_-1/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/output_-1/kernel/mAdam/output_-1/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/output_-1/kernel/vAdam/output_-1/bias/vAdam/output/kernel/vAdam/output/bias/v*#
Tin
2*
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
#__inference__traced_restore_3110246

i
J__inference_dropout_tf_86_layer_call_and_return_conditional_losses_3109730

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№	
п
F__inference_output_-1_layer_call_and_return_conditional_losses_3109749

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я)
 
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109943

inputs,
(output__1_matmul_readvariableop_resource-
)output__1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityЂoutput/BiasAdd/ReadVariableOpЂoutput/MatMul/ReadVariableOpЂ output_-1/BiasAdd/ReadVariableOpЂoutput_-1/MatMul/ReadVariableOp
dropout_tf_86/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout_tf_86/dropout/Const
dropout_tf_86/dropout/MulMulinputs$dropout_tf_86/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_tf_86/dropout/Mulp
dropout_tf_86/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_tf_86/dropout/Shapeо
2dropout_tf_86/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_86/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype024
2dropout_tf_86/dropout/random_uniform/RandomUniform
$dropout_tf_86/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2&
$dropout_tf_86/dropout/GreaterEqual/yі
"dropout_tf_86/dropout/GreaterEqualGreaterEqual;dropout_tf_86/dropout/random_uniform/RandomUniform:output:0-dropout_tf_86/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"dropout_tf_86/dropout/GreaterEqualЉ
dropout_tf_86/dropout/CastCast&dropout_tf_86/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_tf_86/dropout/CastВ
dropout_tf_86/dropout/Mul_1Muldropout_tf_86/dropout/Mul:z:0dropout_tf_86/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_tf_86/dropout/Mul_1Ћ
output_-1/MatMul/ReadVariableOpReadVariableOp(output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
output_-1/MatMul/ReadVariableOpЊ
output_-1/MatMulMatMuldropout_tf_86/dropout/Mul_1:z:0'output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
output_-1/MatMulЊ
 output_-1/BiasAdd/ReadVariableOpReadVariableOp)output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 output_-1/BiasAdd/ReadVariableOpЉ
output_-1/BiasAddBiasAddoutput_-1/MatMul:product:0(output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
output_-1/BiasAddv
output_-1/ReluReluoutput_-1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
output_-1/Relu
dropout_tf_87/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout_tf_87/dropout/ConstГ
dropout_tf_87/dropout/MulMuloutput_-1/Relu:activations:0$dropout_tf_87/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_tf_87/dropout/Mul
dropout_tf_87/dropout/ShapeShapeoutput_-1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_tf_87/dropout/Shapeо
2dropout_tf_87/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_87/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype024
2dropout_tf_87/dropout/random_uniform/RandomUniform
$dropout_tf_87/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2&
$dropout_tf_87/dropout/GreaterEqual/yі
"dropout_tf_87/dropout/GreaterEqualGreaterEqual;dropout_tf_87/dropout/random_uniform/RandomUniform:output:0-dropout_tf_87/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"dropout_tf_87/dropout/GreaterEqualЉ
dropout_tf_87/dropout/CastCast&dropout_tf_87/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
dropout_tf_87/dropout/CastВ
dropout_tf_87/dropout/Mul_1Muldropout_tf_87/dropout/Mul:z:0dropout_tf_87/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_tf_87/dropout/Mul_1Ђ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOpЁ
output/MatMulMatMuldropout_tf_87/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/MatMulЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/BiasAddя
IdentityIdentityoutput/BiasAdd:output:0^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp!^output_-1/BiasAdd/ReadVariableOp ^output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2D
 output_-1/BiasAdd/ReadVariableOp output_-1/BiasAdd/ReadVariableOp2B
output_-1/MatMul/ReadVariableOpoutput_-1/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ф

+__inference_output_-1_layer_call_fn_3110039

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_output_-1_layer_call_and_return_conditional_losses_31097492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

г
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109847

inputs
output__1_3109835
output__1_3109837
output_3109841
output_3109843
identityЂ%dropout_tf_86/StatefulPartitionedCallЂ%dropout_tf_87/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЂ!output_-1/StatefulPartitionedCallџ
%dropout_tf_86/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_tf_86_layer_call_and_return_conditional_losses_31097302'
%dropout_tf_86/StatefulPartitionedCallЧ
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_86/StatefulPartitionedCall:output:0output__1_3109835output__1_3109837*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_output_-1_layer_call_and_return_conditional_losses_31097492#
!output_-1/StatefulPartitionedCallЫ
%dropout_tf_87/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_86/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_tf_87_layer_call_and_return_conditional_losses_31097772'
%dropout_tf_87/StatefulPartitionedCallИ
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_87/StatefulPartitionedCall:output:0output_3109841output_3109843*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_31097952 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_86/StatefulPartitionedCall&^dropout_tf_87/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2N
%dropout_tf_86/StatefulPartitionedCall%dropout_tf_86/StatefulPartitionedCall2N
%dropout_tf_87/StatefulPartitionedCall%dropout_tf_87/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

i
J__inference_dropout_tf_87_layer_call_and_return_conditional_losses_3110051

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ї
 
.__inference_mc_dropout_6_layer_call_fn_3109887	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_31098762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
ї

%__inference_signature_wrapper_3109910	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_31097142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
	
м
C__inference_output_layer_call_and_return_conditional_losses_3110066

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs

i
J__inference_dropout_tf_87_layer_call_and_return_conditional_losses_3109777

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
­
h
/__inference_dropout_tf_87_layer_call_fn_3110056

inputs
identityЂStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_tf_87_layer_call_and_return_conditional_losses_31097772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ222
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
м3
р
"__inference__wrapped_model_3109714	
input9
5mc_dropout_6_output__1_matmul_readvariableop_resource:
6mc_dropout_6_output__1_biasadd_readvariableop_resource6
2mc_dropout_6_output_matmul_readvariableop_resource7
3mc_dropout_6_output_biasadd_readvariableop_resource
identityЂ*mc_dropout_6/output/BiasAdd/ReadVariableOpЂ)mc_dropout_6/output/MatMul/ReadVariableOpЂ-mc_dropout_6/output_-1/BiasAdd/ReadVariableOpЂ,mc_dropout_6/output_-1/MatMul/ReadVariableOp
(mc_dropout_6/dropout_tf_86/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2*
(mc_dropout_6/dropout_tf_86/dropout/ConstУ
&mc_dropout_6/dropout_tf_86/dropout/MulMulinput1mc_dropout_6/dropout_tf_86/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&mc_dropout_6/dropout_tf_86/dropout/Mul
(mc_dropout_6/dropout_tf_86/dropout/ShapeShapeinput*
T0*
_output_shapes
:2*
(mc_dropout_6/dropout_tf_86/dropout/Shape
?mc_dropout_6/dropout_tf_86/dropout/random_uniform/RandomUniformRandomUniform1mc_dropout_6/dropout_tf_86/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02A
?mc_dropout_6/dropout_tf_86/dropout/random_uniform/RandomUniformЋ
1mc_dropout_6/dropout_tf_86/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<23
1mc_dropout_6/dropout_tf_86/dropout/GreaterEqual/yЊ
/mc_dropout_6/dropout_tf_86/dropout/GreaterEqualGreaterEqualHmc_dropout_6/dropout_tf_86/dropout/random_uniform/RandomUniform:output:0:mc_dropout_6/dropout_tf_86/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ21
/mc_dropout_6/dropout_tf_86/dropout/GreaterEqualа
'mc_dropout_6/dropout_tf_86/dropout/CastCast3mc_dropout_6/dropout_tf_86/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2)
'mc_dropout_6/dropout_tf_86/dropout/Castц
(mc_dropout_6/dropout_tf_86/dropout/Mul_1Mul*mc_dropout_6/dropout_tf_86/dropout/Mul:z:0+mc_dropout_6/dropout_tf_86/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(mc_dropout_6/dropout_tf_86/dropout/Mul_1в
,mc_dropout_6/output_-1/MatMul/ReadVariableOpReadVariableOp5mc_dropout_6_output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02.
,mc_dropout_6/output_-1/MatMul/ReadVariableOpо
mc_dropout_6/output_-1/MatMulMatMul,mc_dropout_6/dropout_tf_86/dropout/Mul_1:z:04mc_dropout_6/output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mc_dropout_6/output_-1/MatMulб
-mc_dropout_6/output_-1/BiasAdd/ReadVariableOpReadVariableOp6mc_dropout_6_output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02/
-mc_dropout_6/output_-1/BiasAdd/ReadVariableOpн
mc_dropout_6/output_-1/BiasAddBiasAdd'mc_dropout_6/output_-1/MatMul:product:05mc_dropout_6/output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22 
mc_dropout_6/output_-1/BiasAdd
mc_dropout_6/output_-1/ReluRelu'mc_dropout_6/output_-1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
mc_dropout_6/output_-1/Relu
(mc_dropout_6/dropout_tf_87/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2*
(mc_dropout_6/dropout_tf_87/dropout/Constч
&mc_dropout_6/dropout_tf_87/dropout/MulMul)mc_dropout_6/output_-1/Relu:activations:01mc_dropout_6/dropout_tf_87/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22(
&mc_dropout_6/dropout_tf_87/dropout/Mul­
(mc_dropout_6/dropout_tf_87/dropout/ShapeShape)mc_dropout_6/output_-1/Relu:activations:0*
T0*
_output_shapes
:2*
(mc_dropout_6/dropout_tf_87/dropout/Shape
?mc_dropout_6/dropout_tf_87/dropout/random_uniform/RandomUniformRandomUniform1mc_dropout_6/dropout_tf_87/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype02A
?mc_dropout_6/dropout_tf_87/dropout/random_uniform/RandomUniformЋ
1mc_dropout_6/dropout_tf_87/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<23
1mc_dropout_6/dropout_tf_87/dropout/GreaterEqual/yЊ
/mc_dropout_6/dropout_tf_87/dropout/GreaterEqualGreaterEqualHmc_dropout_6/dropout_tf_87/dropout/random_uniform/RandomUniform:output:0:mc_dropout_6/dropout_tf_87/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ221
/mc_dropout_6/dropout_tf_87/dropout/GreaterEqualа
'mc_dropout_6/dropout_tf_87/dropout/CastCast3mc_dropout_6/dropout_tf_87/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22)
'mc_dropout_6/dropout_tf_87/dropout/Castц
(mc_dropout_6/dropout_tf_87/dropout/Mul_1Mul*mc_dropout_6/dropout_tf_87/dropout/Mul:z:0+mc_dropout_6/dropout_tf_87/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22*
(mc_dropout_6/dropout_tf_87/dropout/Mul_1Щ
)mc_dropout_6/output/MatMul/ReadVariableOpReadVariableOp2mc_dropout_6_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)mc_dropout_6/output/MatMul/ReadVariableOpе
mc_dropout_6/output/MatMulMatMul,mc_dropout_6/dropout_tf_87/dropout/Mul_1:z:01mc_dropout_6/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mc_dropout_6/output/MatMulШ
*mc_dropout_6/output/BiasAdd/ReadVariableOpReadVariableOp3mc_dropout_6_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*mc_dropout_6/output/BiasAdd/ReadVariableOpб
mc_dropout_6/output/BiasAddBiasAdd$mc_dropout_6/output/MatMul:product:02mc_dropout_6/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mc_dropout_6/output/BiasAddА
IdentityIdentity$mc_dropout_6/output/BiasAdd:output:0+^mc_dropout_6/output/BiasAdd/ReadVariableOp*^mc_dropout_6/output/MatMul/ReadVariableOp.^mc_dropout_6/output_-1/BiasAdd/ReadVariableOp-^mc_dropout_6/output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2X
*mc_dropout_6/output/BiasAdd/ReadVariableOp*mc_dropout_6/output/BiasAdd/ReadVariableOp2V
)mc_dropout_6/output/MatMul/ReadVariableOp)mc_dropout_6/output/MatMul/ReadVariableOp2^
-mc_dropout_6/output_-1/BiasAdd/ReadVariableOp-mc_dropout_6/output_-1/BiasAdd/ReadVariableOp2\
,mc_dropout_6/output_-1/MatMul/ReadVariableOp,mc_dropout_6/output_-1/MatMul/ReadVariableOp:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput

в
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109828	
input
output__1_3109816
output__1_3109818
output_3109822
output_3109824
identityЂ%dropout_tf_86/StatefulPartitionedCallЂ%dropout_tf_87/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЂ!output_-1/StatefulPartitionedCallў
%dropout_tf_86/StatefulPartitionedCallStatefulPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_tf_86_layer_call_and_return_conditional_losses_31097302'
%dropout_tf_86/StatefulPartitionedCallЧ
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_86/StatefulPartitionedCall:output:0output__1_3109816output__1_3109818*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_output_-1_layer_call_and_return_conditional_losses_31097492#
!output_-1/StatefulPartitionedCallЫ
%dropout_tf_87/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_86/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_tf_87_layer_call_and_return_conditional_losses_31097772'
%dropout_tf_87/StatefulPartitionedCallИ
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_87/StatefulPartitionedCall:output:0output_3109822output_3109824*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_31097952 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_86/StatefulPartitionedCall&^dropout_tf_87/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2N
%dropout_tf_86/StatefulPartitionedCall%dropout_tf_86/StatefulPartitionedCall2N
%dropout_tf_87/StatefulPartitionedCall%dropout_tf_87/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
Њ
Ё
.__inference_mc_dropout_6_layer_call_fn_3110002

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_31098762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
н
}
(__inference_output_layer_call_fn_3110075

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_31097952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Е5
	
 __inference__traced_save_3110167
file_prefix/
+savev2_output__1_kernel_read_readvariableop-
)savev2_output__1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop6
2savev2_adam_output__1_kernel_m_read_readvariableop4
0savev2_adam_output__1_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop6
2savev2_adam_output__1_kernel_v_read_readvariableop4
0savev2_adam_output__1_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameО
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*а
valueЦBУB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesИ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЇ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_output__1_kernel_read_readvariableop)savev2_output__1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_output__1_kernel_m_read_readvariableop0savev2_adam_output__1_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_output__1_kernel_v_read_readvariableop0savev2_adam_output__1_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes|
z: :2:2:2:: : : : : : : : : : : :2:2:2::2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: 
№	
п
F__inference_output_-1_layer_call_and_return_conditional_losses_3110030

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

i
J__inference_dropout_tf_86_layer_call_and_return_conditional_losses_3110014

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Њ
Ё
.__inference_mc_dropout_6_layer_call_fn_3109989

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_31098472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
м
C__inference_output_layer_call_and_return_conditional_losses_3109795

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ї
 
.__inference_mc_dropout_6_layer_call_fn_3109858	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_31098472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput

г
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109876

inputs
output__1_3109864
output__1_3109866
output_3109870
output_3109872
identityЂ%dropout_tf_86/StatefulPartitionedCallЂ%dropout_tf_87/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЂ!output_-1/StatefulPartitionedCallџ
%dropout_tf_86/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_tf_86_layer_call_and_return_conditional_losses_31097302'
%dropout_tf_86/StatefulPartitionedCallЧ
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_86/StatefulPartitionedCall:output:0output__1_3109864output__1_3109866*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_output_-1_layer_call_and_return_conditional_losses_31097492#
!output_-1/StatefulPartitionedCallЫ
%dropout_tf_87/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_86/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_tf_87_layer_call_and_return_conditional_losses_31097772'
%dropout_tf_87/StatefulPartitionedCallИ
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_87/StatefulPartitionedCall:output:0output_3109870output_3109872*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_31097952 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_86/StatefulPartitionedCall&^dropout_tf_87/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2N
%dropout_tf_86/StatefulPartitionedCall%dropout_tf_86/StatefulPartitionedCall2N
%dropout_tf_87/StatefulPartitionedCall%dropout_tf_87/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я)
 
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109976

inputs,
(output__1_matmul_readvariableop_resource-
)output__1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityЂoutput/BiasAdd/ReadVariableOpЂoutput/MatMul/ReadVariableOpЂ output_-1/BiasAdd/ReadVariableOpЂoutput_-1/MatMul/ReadVariableOp
dropout_tf_86/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout_tf_86/dropout/Const
dropout_tf_86/dropout/MulMulinputs$dropout_tf_86/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_tf_86/dropout/Mulp
dropout_tf_86/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_tf_86/dropout/Shapeо
2dropout_tf_86/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_86/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype024
2dropout_tf_86/dropout/random_uniform/RandomUniform
$dropout_tf_86/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2&
$dropout_tf_86/dropout/GreaterEqual/yі
"dropout_tf_86/dropout/GreaterEqualGreaterEqual;dropout_tf_86/dropout/random_uniform/RandomUniform:output:0-dropout_tf_86/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"dropout_tf_86/dropout/GreaterEqualЉ
dropout_tf_86/dropout/CastCast&dropout_tf_86/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_tf_86/dropout/CastВ
dropout_tf_86/dropout/Mul_1Muldropout_tf_86/dropout/Mul:z:0dropout_tf_86/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_tf_86/dropout/Mul_1Ћ
output_-1/MatMul/ReadVariableOpReadVariableOp(output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
output_-1/MatMul/ReadVariableOpЊ
output_-1/MatMulMatMuldropout_tf_86/dropout/Mul_1:z:0'output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
output_-1/MatMulЊ
 output_-1/BiasAdd/ReadVariableOpReadVariableOp)output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 output_-1/BiasAdd/ReadVariableOpЉ
output_-1/BiasAddBiasAddoutput_-1/MatMul:product:0(output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
output_-1/BiasAddv
output_-1/ReluReluoutput_-1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
output_-1/Relu
dropout_tf_87/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§J?2
dropout_tf_87/dropout/ConstГ
dropout_tf_87/dropout/MulMuloutput_-1/Relu:activations:0$dropout_tf_87/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_tf_87/dropout/Mul
dropout_tf_87/dropout/ShapeShapeoutput_-1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_tf_87/dropout/Shapeо
2dropout_tf_87/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_87/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype024
2dropout_tf_87/dropout/random_uniform/RandomUniform
$dropout_tf_87/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
з#<2&
$dropout_tf_87/dropout/GreaterEqual/yі
"dropout_tf_87/dropout/GreaterEqualGreaterEqual;dropout_tf_87/dropout/random_uniform/RandomUniform:output:0-dropout_tf_87/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22$
"dropout_tf_87/dropout/GreaterEqualЉ
dropout_tf_87/dropout/CastCast&dropout_tf_87/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
dropout_tf_87/dropout/CastВ
dropout_tf_87/dropout/Mul_1Muldropout_tf_87/dropout/Mul:z:0dropout_tf_87/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_tf_87/dropout/Mul_1Ђ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOpЁ
output/MatMulMatMuldropout_tf_87/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/MatMulЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/BiasAddя
IdentityIdentityoutput/BiasAdd:output:0^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp!^output_-1/BiasAdd/ReadVariableOp ^output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2D
 output_-1/BiasAdd/ReadVariableOp output_-1/BiasAdd/ReadVariableOp2B
output_-1/MatMul/ReadVariableOpoutput_-1/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

в
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109812	
input
output__1_3109760
output__1_3109762
output_3109806
output_3109808
identityЂ%dropout_tf_86/StatefulPartitionedCallЂ%dropout_tf_87/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЂ!output_-1/StatefulPartitionedCallў
%dropout_tf_86/StatefulPartitionedCallStatefulPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_tf_86_layer_call_and_return_conditional_losses_31097302'
%dropout_tf_86/StatefulPartitionedCallЧ
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_86/StatefulPartitionedCall:output:0output__1_3109760output__1_3109762*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_output_-1_layer_call_and_return_conditional_losses_31097492#
!output_-1/StatefulPartitionedCallЫ
%dropout_tf_87/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_86/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_tf_87_layer_call_and_return_conditional_losses_31097772'
%dropout_tf_87/StatefulPartitionedCallИ
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_87/StatefulPartitionedCall:output:0output_3109806output_3109808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_31097952 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_86/StatefulPartitionedCall&^dropout_tf_87/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2N
%dropout_tf_86/StatefulPartitionedCall%dropout_tf_86/StatefulPartitionedCall2N
%dropout_tf_87/StatefulPartitionedCall%dropout_tf_87/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
­
h
/__inference_dropout_tf_86_layer_call_fn_3110019

inputs
identityЂStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_dropout_tf_86_layer_call_and_return_conditional_losses_31097302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Эa
Н
#__inference__traced_restore_3110246
file_prefix%
!assignvariableop_output__1_kernel%
!assignvariableop_1_output__1_bias$
 assignvariableop_2_output_kernel"
assignvariableop_3_output_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_1
assignvariableop_13_total_2
assignvariableop_14_count_2/
+assignvariableop_15_adam_output__1_kernel_m-
)assignvariableop_16_adam_output__1_bias_m,
(assignvariableop_17_adam_output_kernel_m*
&assignvariableop_18_adam_output_bias_m/
+assignvariableop_19_adam_output__1_kernel_v-
)assignvariableop_20_adam_output__1_bias_v,
(assignvariableop_21_adam_output_kernel_v*
&assignvariableop_22_adam_output_bias_v
identity_24ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ф
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*а
valueЦBУB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesО
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЃ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_output__1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_output__1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ѕ
AssignVariableOp_2AssignVariableOp assignvariableop_2_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѓ
AssignVariableOp_3AssignVariableOpassignvariableop_3_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4Ё
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѓ
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ѓ
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ђ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Њ
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ё
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ѓ
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ѓ
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ѓ
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ѓ
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Г
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_output__1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Б
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_output__1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17А
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_output_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ў
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_output_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Г
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_output__1_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Б
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_output__1_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21А
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_output_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ў
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_output_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_229
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpи
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23Ы
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2$
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
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ѕ
serving_default
7
input.
serving_default_input:0џџџџџџџџџ:
output0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:№
Ж'
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
_model_paras
comp_args_kw
	optimizer
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
*\&call_and_return_all_conditional_losses
]__call__
^_default_save_signature"в$
_tf_keras_networkЖ${"class_name": "MCDropout", "name": "mc_dropout_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "mc_dropout_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_86", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_tf_86", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_-1", "inbound_nodes": [[["dropout_tf_86", 0, 0, {}]]]}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_87", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_tf_87", "inbound_nodes": [[["output_-1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_tf_87", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 6]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MCDropout", "config": {"name": "mc_dropout_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_86", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_tf_86", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_-1", "inbound_nodes": [[["dropout_tf_86", 0, 0, {}]]]}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_87", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_tf_87", "inbound_nodes": [[["output_-1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_tf_87", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "nll", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
х"т
_tf_keras_input_layerТ{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
ё
trainable_variables
regularization_losses
	variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"т
_tf_keras_layerШ{"class_name": "DropoutTF", "name": "dropout_tf_86", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_tf_86", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}
ђ

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"Э
_tf_keras_layerГ{"class_name": "Dense", "name": "output_-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
ё
trainable_variables
regularization_losses
	variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"т
_tf_keras_layerШ{"class_name": "DropoutTF", "name": "dropout_tf_87", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_tf_87", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}
я

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
*e&call_and_return_all_conditional_losses
f__call__"Ъ
_tf_keras_layerА{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
2
"grid_dropout"
trackable_dict_wrapper
(
#1"
trackable_tuple_wrapper

$iter

%beta_1

&beta_2
	'decay
(learning_ratemTmUmVmWvXvYvZv["
	optimizer
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
Ъ
)layer_regularization_losses
*non_trainable_variables
+metrics
	trainable_variables
,layer_metrics

regularization_losses

-layers
	variables
]__call__
^_default_save_signature
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
,
gserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
.layer_regularization_losses
/non_trainable_variables
0metrics
trainable_variables
1layer_metrics
regularization_losses

2layers
	variables
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
": 22output_-1/kernel
:22output_-1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
3layer_regularization_losses
4non_trainable_variables
5metrics
trainable_variables
6layer_metrics
regularization_losses

7layers
	variables
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
8layer_regularization_losses
9non_trainable_variables
:metrics
trainable_variables
;layer_metrics
regularization_losses

<layers
	variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
:22output/kernel
:2output/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
=layer_regularization_losses
>non_trainable_variables
?metrics
trainable_variables
@layer_metrics
regularization_losses

Alayers
 	variables
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
	optimizer
Bmetrics"
trackable_dict_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Л
	Ftotal
	Gcount
H	variables
I	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
є
	Jtotal
	Kcount
L
_fn_kwargs
M	variables
N	keras_api"­
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
ѓ
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api"Ќ
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
:  (2total
:  (2count
.
F0
G1"
trackable_list_wrapper
-
H	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
':%22Adam/output_-1/kernel/m
!:22Adam/output_-1/bias/m
$:"22Adam/output/kernel/m
:2Adam/output/bias/m
':%22Adam/output_-1/kernel/v
!:22Adam/output_-1/bias/v
$:"22Adam/output/kernel/v
:2Adam/output/bias/v
ђ2я
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109976
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109943
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109828
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109812Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
2
.__inference_mc_dropout_6_layer_call_fn_3109887
.__inference_mc_dropout_6_layer_call_fn_3109989
.__inference_mc_dropout_6_layer_call_fn_3109858
.__inference_mc_dropout_6_layer_call_fn_3110002Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
"__inference__wrapped_model_3109714Д
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *$Ђ!

inputџџџџџџџџџ
є2ё
J__inference_dropout_tf_86_layer_call_and_return_conditional_losses_3110014Ђ
В
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
annotationsЊ *
 
й2ж
/__inference_dropout_tf_86_layer_call_fn_3110019Ђ
В
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
annotationsЊ *
 
№2э
F__inference_output_-1_layer_call_and_return_conditional_losses_3110030Ђ
В
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
annotationsЊ *
 
е2в
+__inference_output_-1_layer_call_fn_3110039Ђ
В
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
annotationsЊ *
 
є2ё
J__inference_dropout_tf_87_layer_call_and_return_conditional_losses_3110051Ђ
В
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
annotationsЊ *
 
й2ж
/__inference_dropout_tf_87_layer_call_fn_3110056Ђ
В
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
annotationsЊ *
 
э2ъ
C__inference_output_layer_call_and_return_conditional_losses_3110066Ђ
В
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
annotationsЊ *
 
в2Я
(__inference_output_layer_call_fn_3110075Ђ
В
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
annotationsЊ *
 
ЪBЧ
%__inference_signature_wrapper_3109910input"
В
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
annotationsЊ *
 
"__inference__wrapped_model_3109714g.Ђ+
$Ђ!

inputџџџџџџџџџ
Њ "/Њ,
*
output 
outputџџџџџџџџџІ
J__inference_dropout_tf_86_layer_call_and_return_conditional_losses_3110014X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
/__inference_dropout_tf_86_layer_call_fn_3110019K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
J__inference_dropout_tf_87_layer_call_and_return_conditional_losses_3110051X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ2
 ~
/__inference_dropout_tf_87_layer_call_fn_3110056K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ2В
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109812e6Ђ3
,Ђ)

inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 В
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109828e6Ђ3
,Ђ)

inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Г
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109943f7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Г
I__inference_mc_dropout_6_layer_call_and_return_conditional_losses_3109976f7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
.__inference_mc_dropout_6_layer_call_fn_3109858X6Ђ3
,Ђ)

inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
.__inference_mc_dropout_6_layer_call_fn_3109887X6Ђ3
,Ђ)

inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
.__inference_mc_dropout_6_layer_call_fn_3109989Y7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
.__inference_mc_dropout_6_layer_call_fn_3110002Y7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџІ
F__inference_output_-1_layer_call_and_return_conditional_losses_3110030\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ2
 ~
+__inference_output_-1_layer_call_fn_3110039O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ2Ѓ
C__inference_output_layer_call_and_return_conditional_losses_3110066\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_output_layer_call_fn_3110075O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ
%__inference_signature_wrapper_3109910p7Ђ4
Ђ 
-Њ*
(
input
inputџџџџџџџџџ"/Њ,
*
output 
outputџџџџџџџџџ