њќ
­
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
8
Const
output"dtype"
valuetensor"
dtypetype
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
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8нЭ

1_output_-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*#
shared_name1_output_-1/kernel
y
&1_output_-1/kernel/Read/ReadVariableOpReadVariableOp1_output_-1/kernel*
_output_shapes

:2*
dtype0
x
1_output_-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*!
shared_name1_output_-1/bias
q
$1_output_-1/bias/Read/ReadVariableOpReadVariableOp1_output_-1/bias*
_output_shapes
:2*
dtype0
z
1_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2* 
shared_name1_output/kernel
s
#1_output/kernel/Read/ReadVariableOpReadVariableOp1_output/kernel*
_output_shapes

:2*
dtype0
r
1_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name1_output/bias
k
!1_output/bias/Read/ReadVariableOpReadVariableOp1_output/bias*
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

Adam/1_output_-1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2**
shared_nameAdam/1_output_-1/kernel/m

-Adam/1_output_-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/1_output_-1/kernel/m*
_output_shapes

:2*
dtype0

Adam/1_output_-1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*(
shared_nameAdam/1_output_-1/bias/m

+Adam/1_output_-1/bias/m/Read/ReadVariableOpReadVariableOpAdam/1_output_-1/bias/m*
_output_shapes
:2*
dtype0

Adam/1_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/1_output/kernel/m

*Adam/1_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/1_output/kernel/m*
_output_shapes

:2*
dtype0

Adam/1_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/1_output/bias/m
y
(Adam/1_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/1_output/bias/m*
_output_shapes
:*
dtype0

Adam/1_output_-1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2**
shared_nameAdam/1_output_-1/kernel/v

-Adam/1_output_-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/1_output_-1/kernel/v*
_output_shapes

:2*
dtype0

Adam/1_output_-1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*(
shared_nameAdam/1_output_-1/bias/v

+Adam/1_output_-1/bias/v/Read/ReadVariableOpReadVariableOpAdam/1_output_-1/bias/v*
_output_shapes
:2*
dtype0

Adam/1_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/1_output/kernel/v

*Adam/1_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/1_output/kernel/v*
_output_shapes

:2*
dtype0

Adam/1_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/1_output/bias/v
y
(Adam/1_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/1_output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ѕ$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*р#
valueж#Bг# BЬ#
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


"tau2
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
^\
VARIABLE_VALUE1_output_-1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE1_output_-1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
[Y
VARIABLE_VALUE1_output/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUE1_output/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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

VARIABLE_VALUEAdam/1_output_-1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/1_output_-1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/1_output/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/1_output/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/1_output_-1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/1_output_-1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/1_output/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/1_output/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_1_inputPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_1_input1_output_-1/kernel1_output_-1/bias1_output/kernel1_output/bias*
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
GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_324470
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&1_output_-1/kernel/Read/ReadVariableOp$1_output_-1/bias/Read/ReadVariableOp#1_output/kernel/Read/ReadVariableOp!1_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp-Adam/1_output_-1/kernel/m/Read/ReadVariableOp+Adam/1_output_-1/bias/m/Read/ReadVariableOp*Adam/1_output/kernel/m/Read/ReadVariableOp(Adam/1_output/bias/m/Read/ReadVariableOp-Adam/1_output_-1/kernel/v/Read/ReadVariableOp+Adam/1_output_-1/bias/v/Read/ReadVariableOp*Adam/1_output/kernel/v/Read/ReadVariableOp(Adam/1_output/bias/v/Read/ReadVariableOpConst*$
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_324733
Б
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename1_output_-1/kernel1_output_-1/bias1_output/kernel1_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/1_output_-1/kernel/mAdam/1_output_-1/bias/mAdam/1_output/kernel/mAdam/1_output/bias/mAdam/1_output_-1/kernel/vAdam/1_output_-1/bias/vAdam/1_output/kernel/vAdam/1_output/bias/v*#
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_324812дь

f
G__inference_dropout_295_layer_call_and_return_conditional_losses_324607

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
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
 *    2
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

f
G__inference_dropout_294_layer_call_and_return_conditional_losses_324560

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


'__inference_nlm_13_layer_call_fn_324535

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *K
fFRD
B__inference_nlm_13_layer_call_and_return_conditional_losses_3244072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ
e
G__inference_dropout_295_layer_call_and_return_conditional_losses_324612

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs


'__inference_nlm_13_layer_call_fn_324447	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *K
fFRD
B__inference_nlm_13_layer_call_and_return_conditional_losses_3244362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	1_input

H
,__inference_dropout_295_layer_call_fn_324622

inputs
identityШ
PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_295_layer_call_and_return_conditional_losses_3243322
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
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

f
G__inference_dropout_294_layer_call_and_return_conditional_losses_324270

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
н
D__inference_1_output_layer_call_and_return_conditional_losses_324355

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
ё	
р
G__inference_1_output_-1_layer_call_and_return_conditional_losses_324586

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
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
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р)
Ё
B__inference_nlm_13_layer_call_and_return_conditional_losses_324503

inputs,
(output__1_matmul_readvariableop_resource-
)output__1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityЂ1_output/BiasAdd/ReadVariableOpЂ1_output/MatMul/ReadVariableOpЂ"1_output_-1/BiasAdd/ReadVariableOpЂ!1_output_-1/MatMul/ReadVariableOp{
dropout_294/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_294/dropout/Const
dropout_294/dropout/MulMulinputs"dropout_294/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_294/dropout/Mull
dropout_294/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_294/dropout/Shapeи
0dropout_294/dropout/random_uniform/RandomUniformRandomUniform"dropout_294/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype022
0dropout_294/dropout/random_uniform/RandomUniform
"dropout_294/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout_294/dropout/GreaterEqual/yю
 dropout_294/dropout/GreaterEqualGreaterEqual9dropout_294/dropout/random_uniform/RandomUniform:output:0+dropout_294/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 dropout_294/dropout/GreaterEqualЃ
dropout_294/dropout/CastCast$dropout_294/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_294/dropout/CastЊ
dropout_294/dropout/Mul_1Muldropout_294/dropout/Mul:z:0dropout_294/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_294/dropout/Mul_1Џ
!1_output_-1/MatMul/ReadVariableOpReadVariableOp(output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02#
!1_output_-1/MatMul/ReadVariableOpЎ
1_output_-1/MatMulMatMuldropout_294/dropout/Mul_1:z:0)1_output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
1_output_-1/MatMulЎ
"1_output_-1/BiasAdd/ReadVariableOpReadVariableOp)output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02$
"1_output_-1/BiasAdd/ReadVariableOpБ
1_output_-1/BiasAddBiasAdd1_output_-1/MatMul:product:0*1_output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
1_output_-1/BiasAdd|
1_output_-1/ReluRelu1_output_-1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
1_output_-1/Relu{
dropout_295/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_295/dropout/ConstЏ
dropout_295/dropout/MulMul1_output_-1/Relu:activations:0"dropout_295/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_295/dropout/Mul
dropout_295/dropout/ShapeShape1_output_-1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_295/dropout/Shapeи
0dropout_295/dropout/random_uniform/RandomUniformRandomUniform"dropout_295/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
dtype022
0dropout_295/dropout/random_uniform/RandomUniform
"dropout_295/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout_295/dropout/GreaterEqual/yю
 dropout_295/dropout/GreaterEqualGreaterEqual9dropout_295/dropout/random_uniform/RandomUniform:output:0+dropout_295/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22"
 dropout_295/dropout/GreaterEqualЃ
dropout_295/dropout/CastCast$dropout_295/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
dropout_295/dropout/CastЊ
dropout_295/dropout/Mul_1Muldropout_295/dropout/Mul:z:0dropout_295/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_295/dropout/Mul_1І
1_output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02 
1_output/MatMul/ReadVariableOpЅ
1_output/MatMulMatMuldropout_295/dropout/Mul_1:z:0&1_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
1_output/MatMulЅ
1_output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
1_output/BiasAdd/ReadVariableOpЅ
1_output/BiasAddBiasAdd1_output/MatMul:product:0'1_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
1_output/BiasAddљ
IdentityIdentity1_output/BiasAdd:output:0 ^1_output/BiasAdd/ReadVariableOp^1_output/MatMul/ReadVariableOp#^1_output_-1/BiasAdd/ReadVariableOp"^1_output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2B
1_output/BiasAdd/ReadVariableOp1_output/BiasAdd/ReadVariableOp2@
1_output/MatMul/ReadVariableOp1_output/MatMul/ReadVariableOp2H
"1_output_-1/BiasAdd/ReadVariableOp"1_output_-1/BiasAdd/ReadVariableOp2F
!1_output_-1/MatMul/ReadVariableOp!1_output_-1/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц
П
!__inference__wrapped_model_324254	
input5
1nlm_13_1_output__1_matmul_readvariableop_resource6
2nlm_13_1_output__1_biasadd_readvariableop_resource2
.nlm_13_1_output_matmul_readvariableop_resource3
/nlm_13_1_output_biasadd_readvariableop_resource
identityЂ&nlm_13/1_output/BiasAdd/ReadVariableOpЂ%nlm_13/1_output/MatMul/ReadVariableOpЂ)nlm_13/1_output_-1/BiasAdd/ReadVariableOpЂ(nlm_13/1_output_-1/MatMul/ReadVariableOp
nlm_13/dropout_294/IdentityIdentityinput*
T0*'
_output_shapes
:џџџџџџџџџ2
nlm_13/dropout_294/IdentityЦ
(nlm_13/1_output_-1/MatMul/ReadVariableOpReadVariableOp1nlm_13_1_output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02*
(nlm_13/1_output_-1/MatMul/ReadVariableOpЪ
nlm_13/1_output_-1/MatMulMatMul$nlm_13/dropout_294/Identity:output:00nlm_13/1_output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
nlm_13/1_output_-1/MatMulХ
)nlm_13/1_output_-1/BiasAdd/ReadVariableOpReadVariableOp2nlm_13_1_output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)nlm_13/1_output_-1/BiasAdd/ReadVariableOpЭ
nlm_13/1_output_-1/BiasAddBiasAdd#nlm_13/1_output_-1/MatMul:product:01nlm_13/1_output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
nlm_13/1_output_-1/BiasAdd
nlm_13/1_output_-1/ReluRelu#nlm_13/1_output_-1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
nlm_13/1_output_-1/Relu
nlm_13/dropout_295/IdentityIdentity%nlm_13/1_output_-1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
nlm_13/dropout_295/IdentityН
%nlm_13/1_output/MatMul/ReadVariableOpReadVariableOp.nlm_13_1_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02'
%nlm_13/1_output/MatMul/ReadVariableOpС
nlm_13/1_output/MatMulMatMul$nlm_13/dropout_295/Identity:output:0-nlm_13/1_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
nlm_13/1_output/MatMulМ
&nlm_13/1_output/BiasAdd/ReadVariableOpReadVariableOp/nlm_13_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&nlm_13/1_output/BiasAdd/ReadVariableOpС
nlm_13/1_output/BiasAddBiasAdd nlm_13/1_output/MatMul:product:0.nlm_13/1_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
nlm_13/1_output/BiasAdd
IdentityIdentity nlm_13/1_output/BiasAdd:output:0'^nlm_13/1_output/BiasAdd/ReadVariableOp&^nlm_13/1_output/MatMul/ReadVariableOp*^nlm_13/1_output_-1/BiasAdd/ReadVariableOp)^nlm_13/1_output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2P
&nlm_13/1_output/BiasAdd/ReadVariableOp&nlm_13/1_output/BiasAdd/ReadVariableOp2N
%nlm_13/1_output/MatMul/ReadVariableOp%nlm_13/1_output/MatMul/ReadVariableOp2V
)nlm_13/1_output_-1/BiasAdd/ReadVariableOp)nlm_13/1_output_-1/BiasAdd/ReadVariableOp2T
(nlm_13/1_output_-1/MatMul/ReadVariableOp(nlm_13/1_output_-1/MatMul/ReadVariableOp:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	1_input
њ
Ч
B__inference_nlm_13_layer_call_and_return_conditional_losses_324372	
input
output__1_324310
output__1_324312
output_324366
output_324368
identityЂ 1_output/StatefulPartitionedCallЂ#1_output_-1/StatefulPartitionedCallЂ#dropout_294/StatefulPartitionedCallЂ#dropout_295/StatefulPartitionedCallї
#dropout_294/StatefulPartitionedCallStatefulPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_294_layer_call_and_return_conditional_losses_3242702%
#dropout_294/StatefulPartitionedCallШ
#1_output_-1/StatefulPartitionedCallStatefulPartitionedCall,dropout_294/StatefulPartitionedCall:output:0output__1_324310output__1_324312*
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
GPU2*0J 8 *P
fKRI
G__inference_1_output_-1_layer_call_and_return_conditional_losses_3242992%
#1_output_-1/StatefulPartitionedCallФ
#dropout_295/StatefulPartitionedCallStatefulPartitionedCall,1_output_-1/StatefulPartitionedCall:output:0$^dropout_294/StatefulPartitionedCall*
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_295_layer_call_and_return_conditional_losses_3243272%
#dropout_295/StatefulPartitionedCallЙ
 1_output/StatefulPartitionedCallStatefulPartitionedCall,dropout_295/StatefulPartitionedCall:output:0output_324366output_324368*
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
GPU2*0J 8 *M
fHRF
D__inference_1_output_layer_call_and_return_conditional_losses_3243552"
 1_output/StatefulPartitionedCall
IdentityIdentity)1_output/StatefulPartitionedCall:output:0!^1_output/StatefulPartitionedCall$^1_output_-1/StatefulPartitionedCall$^dropout_294/StatefulPartitionedCall$^dropout_295/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2D
 1_output/StatefulPartitionedCall 1_output/StatefulPartitionedCall2J
#1_output_-1/StatefulPartitionedCall#1_output_-1/StatefulPartitionedCall2J
#dropout_294/StatefulPartitionedCall#dropout_294/StatefulPartitionedCall2J
#dropout_295/StatefulPartitionedCall#dropout_295/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	1_input
Ъ
e
G__inference_dropout_294_layer_call_and_return_conditional_losses_324275

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

H
,__inference_dropout_294_layer_call_fn_324575

inputs
identityШ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_294_layer_call_and_return_conditional_losses_3242752
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ
e
G__inference_dropout_294_layer_call_and_return_conditional_losses_324565

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї
e
,__inference_dropout_294_layer_call_fn_324570

inputs
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_294_layer_call_and_return_conditional_losses_3242702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц

,__inference_1_output_-1_layer_call_fn_324595

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
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
GPU2*0J 8 *P
fKRI
G__inference_1_output_-1_layer_call_and_return_conditional_losses_3242992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ї

$__inference_signature_wrapper_324470	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallэ
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
GPU2*0J 8 **
f%R#
!__inference__wrapped_model_3242542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	1_input


'__inference_nlm_13_layer_call_fn_324418	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *K
fFRD
B__inference_nlm_13_layer_call_and_return_conditional_losses_3244072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	1_input
ћ
Ш
B__inference_nlm_13_layer_call_and_return_conditional_losses_324407

inputs
output__1_324395
output__1_324397
output_324401
output_324403
identityЂ 1_output/StatefulPartitionedCallЂ#1_output_-1/StatefulPartitionedCallЂ#dropout_294/StatefulPartitionedCallЂ#dropout_295/StatefulPartitionedCallј
#dropout_294/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_294_layer_call_and_return_conditional_losses_3242702%
#dropout_294/StatefulPartitionedCallШ
#1_output_-1/StatefulPartitionedCallStatefulPartitionedCall,dropout_294/StatefulPartitionedCall:output:0output__1_324395output__1_324397*
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
GPU2*0J 8 *P
fKRI
G__inference_1_output_-1_layer_call_and_return_conditional_losses_3242992%
#1_output_-1/StatefulPartitionedCallФ
#dropout_295/StatefulPartitionedCallStatefulPartitionedCall,1_output_-1/StatefulPartitionedCall:output:0$^dropout_294/StatefulPartitionedCall*
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_295_layer_call_and_return_conditional_losses_3243272%
#dropout_295/StatefulPartitionedCallЙ
 1_output/StatefulPartitionedCallStatefulPartitionedCall,dropout_295/StatefulPartitionedCall:output:0output_324401output_324403*
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
GPU2*0J 8 *M
fHRF
D__inference_1_output_layer_call_and_return_conditional_losses_3243552"
 1_output/StatefulPartitionedCall
IdentityIdentity)1_output/StatefulPartitionedCall:output:0!^1_output/StatefulPartitionedCall$^1_output_-1/StatefulPartitionedCall$^dropout_294/StatefulPartitionedCall$^dropout_295/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2D
 1_output/StatefulPartitionedCall 1_output/StatefulPartitionedCall2J
#1_output_-1/StatefulPartitionedCall#1_output_-1/StatefulPartitionedCall2J
#dropout_294/StatefulPartitionedCall#dropout_294/StatefulPartitionedCall2J
#dropout_295/StatefulPartitionedCall#dropout_295/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ф5
Ж	
__inference__traced_save_324733
file_prefix1
-savev2_1_output__1_kernel_read_readvariableop/
+savev2_1_output__1_bias_read_readvariableop.
*savev2_1_output_kernel_read_readvariableop,
(savev2_1_output_bias_read_readvariableop(
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
"savev2_count_2_read_readvariableop8
4savev2_adam_1_output__1_kernel_m_read_readvariableop6
2savev2_adam_1_output__1_bias_m_read_readvariableop5
1savev2_adam_1_output_kernel_m_read_readvariableop3
/savev2_adam_1_output_bias_m_read_readvariableop8
4savev2_adam_1_output__1_kernel_v_read_readvariableop6
2savev2_adam_1_output__1_bias_v_read_readvariableop5
1savev2_adam_1_output_kernel_v_read_readvariableop3
/savev2_adam_1_output_bias_v_read_readvariableop
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
SaveV2/shape_and_slicesП	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_1_output__1_kernel_read_readvariableop+savev2_1_output__1_bias_read_readvariableop*savev2_1_output_kernel_read_readvariableop(savev2_1_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop4savev2_adam_1_output__1_kernel_m_read_readvariableop2savev2_adam_1_output__1_bias_m_read_readvariableop1savev2_adam_1_output_kernel_m_read_readvariableop/savev2_adam_1_output_bias_m_read_readvariableop4savev2_adam_1_output__1_kernel_v_read_readvariableop2savev2_adam_1_output__1_bias_v_read_readvariableop1savev2_adam_1_output_kernel_v_read_readvariableop/savev2_adam_1_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
z: :2:2:2:: : : : : : : : : : : :2:2:2::2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 
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

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:2: 
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
ф
ћ
B__inference_nlm_13_layer_call_and_return_conditional_losses_324388	
input
output__1_324376
output__1_324378
output_324382
output_324384
identityЂ 1_output/StatefulPartitionedCallЂ#1_output_-1/StatefulPartitionedCallп
dropout_294/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_294_layer_call_and_return_conditional_losses_3242752
dropout_294/PartitionedCallР
#1_output_-1/StatefulPartitionedCallStatefulPartitionedCall$dropout_294/PartitionedCall:output:0output__1_324376output__1_324378*
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
GPU2*0J 8 *P
fKRI
G__inference_1_output_-1_layer_call_and_return_conditional_losses_3242992%
#1_output_-1/StatefulPartitionedCall
dropout_295/PartitionedCallPartitionedCall,1_output_-1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_295_layer_call_and_return_conditional_losses_3243322
dropout_295/PartitionedCallБ
 1_output/StatefulPartitionedCallStatefulPartitionedCall$dropout_295/PartitionedCall:output:0output_324382output_324384*
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
GPU2*0J 8 *M
fHRF
D__inference_1_output_layer_call_and_return_conditional_losses_3243552"
 1_output/StatefulPartitionedCallЦ
IdentityIdentity)1_output/StatefulPartitionedCall:output:0!^1_output/StatefulPartitionedCall$^1_output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2D
 1_output/StatefulPartitionedCall 1_output/StatefulPartitionedCall2J
#1_output_-1/StatefulPartitionedCall#1_output_-1/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	1_input

f
G__inference_dropout_295_layer_call_and_return_conditional_losses_324327

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
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
 *    2
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
п
~
)__inference_1_output_layer_call_fn_324641

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallї
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
GPU2*0J 8 *M
fHRF
D__inference_1_output_layer_call_and_return_conditional_losses_3243552
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


'__inference_nlm_13_layer_call_fn_324548

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *K
fFRD
B__inference_nlm_13_layer_call_and_return_conditional_losses_3244362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
н
D__inference_1_output_layer_call_and_return_conditional_losses_324632

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
Ш
Ё
B__inference_nlm_13_layer_call_and_return_conditional_losses_324522

inputs,
(output__1_matmul_readvariableop_resource-
)output__1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityЂ1_output/BiasAdd/ReadVariableOpЂ1_output/MatMul/ReadVariableOpЂ"1_output_-1/BiasAdd/ReadVariableOpЂ!1_output_-1/MatMul/ReadVariableOpr
dropout_294/IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_294/IdentityЏ
!1_output_-1/MatMul/ReadVariableOpReadVariableOp(output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02#
!1_output_-1/MatMul/ReadVariableOpЎ
1_output_-1/MatMulMatMuldropout_294/Identity:output:0)1_output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
1_output_-1/MatMulЎ
"1_output_-1/BiasAdd/ReadVariableOpReadVariableOp)output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02$
"1_output_-1/BiasAdd/ReadVariableOpБ
1_output_-1/BiasAddBiasAdd1_output_-1/MatMul:product:0*1_output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ22
1_output_-1/BiasAdd|
1_output_-1/ReluRelu1_output_-1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22
1_output_-1/Relu
dropout_295/IdentityIdentity1_output_-1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ22
dropout_295/IdentityІ
1_output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02 
1_output/MatMul/ReadVariableOpЅ
1_output/MatMulMatMuldropout_295/Identity:output:0&1_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
1_output/MatMulЅ
1_output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
1_output/BiasAdd/ReadVariableOpЅ
1_output/BiasAddBiasAdd1_output/MatMul:product:0'1_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
1_output/BiasAddљ
IdentityIdentity1_output/BiasAdd:output:0 ^1_output/BiasAdd/ReadVariableOp^1_output/MatMul/ReadVariableOp#^1_output_-1/BiasAdd/ReadVariableOp"^1_output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2B
1_output/BiasAdd/ReadVariableOp1_output/BiasAdd/ReadVariableOp2@
1_output/MatMul/ReadVariableOp1_output/MatMul/ReadVariableOp2H
"1_output_-1/BiasAdd/ReadVariableOp"1_output_-1/BiasAdd/ReadVariableOp2F
!1_output_-1/MatMul/ReadVariableOp!1_output_-1/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ
e
G__inference_dropout_295_layer_call_and_return_conditional_losses_324332

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ2:O K
'
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
Ї
e
,__inference_dropout_295_layer_call_fn_324617

inputs
identityЂStatefulPartitionedCallр
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_295_layer_call_and_return_conditional_losses_3243272
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
х
ќ
B__inference_nlm_13_layer_call_and_return_conditional_losses_324436

inputs
output__1_324424
output__1_324426
output_324430
output_324432
identityЂ 1_output/StatefulPartitionedCallЂ#1_output_-1/StatefulPartitionedCallр
dropout_294/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_dropout_294_layer_call_and_return_conditional_losses_3242752
dropout_294/PartitionedCallР
#1_output_-1/StatefulPartitionedCallStatefulPartitionedCall$dropout_294/PartitionedCall:output:0output__1_324424output__1_324426*
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
GPU2*0J 8 *P
fKRI
G__inference_1_output_-1_layer_call_and_return_conditional_losses_3242992%
#1_output_-1/StatefulPartitionedCall
dropout_295/PartitionedCallPartitionedCall,1_output_-1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8 *P
fKRI
G__inference_dropout_295_layer_call_and_return_conditional_losses_3243322
dropout_295/PartitionedCallБ
 1_output/StatefulPartitionedCallStatefulPartitionedCall$dropout_295/PartitionedCall:output:0output_324430output_324432*
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
GPU2*0J 8 *M
fHRF
D__inference_1_output_layer_call_and_return_conditional_losses_3243552"
 1_output/StatefulPartitionedCallЦ
IdentityIdentity)1_output/StatefulPartitionedCall:output:0!^1_output/StatefulPartitionedCall$^1_output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2D
 1_output/StatefulPartitionedCall 1_output/StatefulPartitionedCall2J
#1_output_-1/StatefulPartitionedCall#1_output_-1/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ё	
р
G__inference_1_output_-1_layer_call_and_return_conditional_losses_324299

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
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
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ќa
д
"__inference__traced_restore_324812
file_prefix'
#assignvariableop_1_output__1_kernel'
#assignvariableop_1_1_output__1_bias&
"assignvariableop_2_1_output_kernel$
 assignvariableop_3_1_output_bias 
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
assignvariableop_14_count_21
-assignvariableop_15_adam_1_output__1_kernel_m/
+assignvariableop_16_adam_1_output__1_bias_m.
*assignvariableop_17_adam_1_output_kernel_m,
(assignvariableop_18_adam_1_output_bias_m1
-assignvariableop_19_adam_1_output__1_kernel_v/
+assignvariableop_20_adam_1_output__1_bias_v.
*assignvariableop_21_adam_1_output_kernel_v,
(assignvariableop_22_adam_1_output_bias_v
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

IdentityЂ
AssignVariableOpAssignVariableOp#assignvariableop_1_output__1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ј
AssignVariableOp_1AssignVariableOp#assignvariableop_1_1_output__1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_1_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_1_output_biasIdentity_3:output:0"/device:CPU:0*
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
Identity_15Е
AssignVariableOp_15AssignVariableOp-assignvariableop_15_adam_1_output__1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Г
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_1_output__1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17В
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_1_output_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18А
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_1_output_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Е
AssignVariableOp_19AssignVariableOp-assignvariableop_19_adam_1_output__1_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Г
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_1_output__1_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21В
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_1_output_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_1_output_bias_vIdentity_22:output:0"/device:CPU:0*
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

NoOp*Ћ
serving_default
;
1_input0
serving_default_1_input:0џџџџџџџџџ<
1_output0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:
'
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
^_default_save_signature"А$
_tf_keras_network${"class_name": "NLM", "name": "nlm_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "nlm_13", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "1_input"}, "name": "1_input", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_294", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}, "name": "dropout_294", "inbound_nodes": [[["1_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "1_output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "1_output_-1", "inbound_nodes": [[["dropout_294", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_295", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}, "name": "dropout_295", "inbound_nodes": [[["1_output_-1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "1_output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "1_output", "inbound_nodes": [[["dropout_295", 0, 0, {}]]]}], "input_layers": [["1_input", 0, 0]], "output_layers": [["1_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "NLM", "config": {"name": "nlm_13", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "1_input"}, "name": "1_input", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_294", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}, "name": "dropout_294", "inbound_nodes": [[["1_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "1_output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "1_output_-1", "inbound_nodes": [[["dropout_294", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_295", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}, "name": "dropout_295", "inbound_nodes": [[["1_output_-1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "1_output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "1_output", "inbound_nodes": [[["dropout_295", 0, 0, {}]]]}], "input_layers": [["1_input", 0, 0]], "output_layers": [["1_output", 0, 0]]}}, "training_config": {"loss": "nll", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
щ"ц
_tf_keras_input_layerЦ{"class_name": "InputLayer", "name": "1_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "1_input"}}
ч
trainable_variables
regularization_losses
	variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_294", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_294", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}}
і

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"б
_tf_keras_layerЗ{"class_name": "Dense", "name": "1_output_-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "1_output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
ч
trainable_variables
regularization_losses
	variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_295", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_295", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}}
ѓ

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
*e&call_and_return_all_conditional_losses
f__call__"Ю
_tf_keras_layerД{"class_name": "Dense", "name": "1_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "1_output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
*
"tau2"
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
$:"221_output_-1/kernel
:221_output_-1/bias
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
!:221_output/kernel
:21_output/bias
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
):'22Adam/1_output_-1/kernel/m
#:!22Adam/1_output_-1/bias/m
&:$22Adam/1_output/kernel/m
 :2Adam/1_output/bias/m
):'22Adam/1_output_-1/kernel/v
#:!22Adam/1_output_-1/bias/v
&:$22Adam/1_output/kernel/v
 :2Adam/1_output/bias/v
ж2г
B__inference_nlm_13_layer_call_and_return_conditional_losses_324503
B__inference_nlm_13_layer_call_and_return_conditional_losses_324522
B__inference_nlm_13_layer_call_and_return_conditional_losses_324372
B__inference_nlm_13_layer_call_and_return_conditional_losses_324388Р
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
ъ2ч
'__inference_nlm_13_layer_call_fn_324548
'__inference_nlm_13_layer_call_fn_324418
'__inference_nlm_13_layer_call_fn_324535
'__inference_nlm_13_layer_call_fn_324447Р
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
п2м
!__inference__wrapped_model_324254Ж
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
annotationsЊ *&Ђ#
!
1_inputџџџџџџџџџ
Ь2Щ
G__inference_dropout_294_layer_call_and_return_conditional_losses_324565
G__inference_dropout_294_layer_call_and_return_conditional_losses_324560Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
,__inference_dropout_294_layer_call_fn_324570
,__inference_dropout_294_layer_call_fn_324575Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ё2ю
G__inference_1_output_-1_layer_call_and_return_conditional_losses_324586Ђ
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
ж2г
,__inference_1_output_-1_layer_call_fn_324595Ђ
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
Ь2Щ
G__inference_dropout_295_layer_call_and_return_conditional_losses_324612
G__inference_dropout_295_layer_call_and_return_conditional_losses_324607Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
,__inference_dropout_295_layer_call_fn_324617
,__inference_dropout_295_layer_call_fn_324622Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
D__inference_1_output_layer_call_and_return_conditional_losses_324632Ђ
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
г2а
)__inference_1_output_layer_call_fn_324641Ђ
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
ЫBШ
$__inference_signature_wrapper_3244701_input"
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
 Ї
G__inference_1_output_-1_layer_call_and_return_conditional_losses_324586\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ2
 
,__inference_1_output_-1_layer_call_fn_324595O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ2Є
D__inference_1_output_layer_call_and_return_conditional_losses_324632\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_1_output_layer_call_fn_324641O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ
!__inference__wrapped_model_324254m0Ђ-
&Ђ#
!
1_inputџџџџџџџџџ
Њ "3Њ0
.
1_output"
1_outputџџџџџџџџџЇ
G__inference_dropout_294_layer_call_and_return_conditional_losses_324560\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 Ї
G__inference_dropout_294_layer_call_and_return_conditional_losses_324565\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 
,__inference_dropout_294_layer_call_fn_324570O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ
,__inference_dropout_294_layer_call_fn_324575O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџЇ
G__inference_dropout_295_layer_call_and_return_conditional_losses_324607\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p
Њ "%Ђ"

0џџџџџџџџџ2
 Ї
G__inference_dropout_295_layer_call_and_return_conditional_losses_324612\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p 
Њ "%Ђ"

0џџџџџџџџџ2
 
,__inference_dropout_295_layer_call_fn_324617O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p
Њ "џџџџџџџџџ2
,__inference_dropout_295_layer_call_fn_324622O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ2
p 
Њ "џџџџџџџџџ2­
B__inference_nlm_13_layer_call_and_return_conditional_losses_324372g8Ђ5
.Ђ+
!
1_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 ­
B__inference_nlm_13_layer_call_and_return_conditional_losses_324388g8Ђ5
.Ђ+
!
1_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ќ
B__inference_nlm_13_layer_call_and_return_conditional_losses_324503f7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ќ
B__inference_nlm_13_layer_call_and_return_conditional_losses_324522f7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
'__inference_nlm_13_layer_call_fn_324418Z8Ђ5
.Ђ+
!
1_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
'__inference_nlm_13_layer_call_fn_324447Z8Ђ5
.Ђ+
!
1_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
'__inference_nlm_13_layer_call_fn_324535Y7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
'__inference_nlm_13_layer_call_fn_324548Y7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ 
$__inference_signature_wrapper_324470x;Ђ8
Ђ 
1Њ.
,
1_input!
1_inputџџџџџџџџџ"3Њ0
.
1_output"
1_outputџџџџџџџџџ