
Ò¨
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
¾
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
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8ç
|
output_-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*!
shared_nameoutput_-1/kernel
u
$output_-1/kernel/Read/ReadVariableOpReadVariableOpoutput_-1/kernel*
_output_shapes

:2*
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
:2*(
shared_nameAdam/output_-1/kernel/m

+Adam/output_-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_-1/kernel/m*
_output_shapes

:2*
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
:2*(
shared_nameAdam/output_-1/kernel/v

+Adam/output_-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_-1/kernel/v*
_output_shapes

:2*
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
dtype0*Î#
valueÄ#BÁ# Bº#
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
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ù
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputoutput_-1/kerneloutput_-1/biasoutput/kerneloutput/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_917010
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ê
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_917267

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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_917346¾
ï	
Þ
E__inference_output_-1_layer_call_and_return_conditional_losses_917130

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
I__inference_dropout_tf_44_layer_call_and_return_conditional_losses_917114

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ª¤?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ

$__inference_signature_wrapper_917010	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_9168142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput

Î
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_916976

inputs
output__1_916964
output__1_916966
output_916970
output_916972
identity¢%dropout_tf_44/StatefulPartitionedCall¢%dropout_tf_45/StatefulPartitionedCall¢output/StatefulPartitionedCall¢!output_-1/StatefulPartitionedCallþ
%dropout_tf_44/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dropout_tf_44_layer_call_and_return_conditional_losses_9168302'
%dropout_tf_44/StatefulPartitionedCallÄ
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_44/StatefulPartitionedCall:output:0output__1_916964output__1_916966*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_-1_layer_call_and_return_conditional_losses_9168492#
!output_-1/StatefulPartitionedCallÊ
%dropout_tf_45/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_44/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dropout_tf_45_layer_call_and_return_conditional_losses_9168772'
%dropout_tf_45/StatefulPartitionedCallµ
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_45/StatefulPartitionedCall:output:0output_916970output_916972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9168952 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_44/StatefulPartitionedCall&^dropout_tf_45/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2N
%dropout_tf_44/StatefulPartitionedCall%dropout_tf_44/StatefulPartitionedCall2N
%dropout_tf_45/StatefulPartitionedCall%dropout_tf_45/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ìa
¼
"__inference__traced_restore_917346
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
identity_24¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ä
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ð
valueÆBÃB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¾
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices£
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

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_output__1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¥
AssignVariableOp_2AssignVariableOp assignvariableop_2_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4¡
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¢
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ª
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
Identity_10¡
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11£
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12£
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14£
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15³
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_output__1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_output__1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17°
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_output_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18®
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_output_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19³
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_output__1_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20±
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_output__1_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21°
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_output_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22®
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_output_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_229
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpØ
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23Ë
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
_user_specified_namefile_prefix
	
Û
B__inference_output_layer_call_and_return_conditional_losses_917166

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
¨
 
-__inference_mc_dropout_3_layer_call_fn_917102

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_9169762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
I__inference_dropout_tf_45_layer_call_and_return_conditional_losses_917151

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ª¤?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
¥

-__inference_mc_dropout_3_layer_call_fn_916958	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_9169472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput

h
I__inference_dropout_tf_44_layer_call_and_return_conditional_losses_916830

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ª¤?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Í
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_916928	
input
output__1_916916
output__1_916918
output_916922
output_916924
identity¢%dropout_tf_44/StatefulPartitionedCall¢%dropout_tf_45/StatefulPartitionedCall¢output/StatefulPartitionedCall¢!output_-1/StatefulPartitionedCallý
%dropout_tf_44/StatefulPartitionedCallStatefulPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dropout_tf_44_layer_call_and_return_conditional_losses_9168302'
%dropout_tf_44/StatefulPartitionedCallÄ
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_44/StatefulPartitionedCall:output:0output__1_916916output__1_916918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_-1_layer_call_and_return_conditional_losses_9168492#
!output_-1/StatefulPartitionedCallÊ
%dropout_tf_45/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_44/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dropout_tf_45_layer_call_and_return_conditional_losses_9168772'
%dropout_tf_45/StatefulPartitionedCallµ
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_45/StatefulPartitionedCall:output:0output_916922output_916924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9168952 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_44/StatefulPartitionedCall&^dropout_tf_45/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2N
%dropout_tf_44/StatefulPartitionedCall%dropout_tf_44/StatefulPartitionedCall2N
%dropout_tf_45/StatefulPartitionedCall%dropout_tf_45/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Û
|
'__inference_output_layer_call_fn_917175

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9168952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
¨
 
-__inference_mc_dropout_3_layer_call_fn_917089

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_9169472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Û
B__inference_output_layer_call_and_return_conditional_losses_916895

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
¥

-__inference_mc_dropout_3_layer_call_fn_916987	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_9169762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
«
g
.__inference_dropout_tf_44_layer_call_fn_917119

inputs
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dropout_tf_44_layer_call_and_return_conditional_losses_9168302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´5
	
__inference__traced_save_917267
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

identity_1¢MergeV2Checkpoints
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¾
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ð
valueÆBÃB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¸
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices§	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_output__1_kernel_read_readvariableop)savev2_output__1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_output__1_kernel_m_read_readvariableop0savev2_adam_output__1_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_output__1_kernel_v_read_readvariableop0savev2_adam_output__1_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

Î
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_916947

inputs
output__1_916935
output__1_916937
output_916941
output_916943
identity¢%dropout_tf_44/StatefulPartitionedCall¢%dropout_tf_45/StatefulPartitionedCall¢output/StatefulPartitionedCall¢!output_-1/StatefulPartitionedCallþ
%dropout_tf_44/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dropout_tf_44_layer_call_and_return_conditional_losses_9168302'
%dropout_tf_44/StatefulPartitionedCallÄ
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_44/StatefulPartitionedCall:output:0output__1_916935output__1_916937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_-1_layer_call_and_return_conditional_losses_9168492#
!output_-1/StatefulPartitionedCallÊ
%dropout_tf_45/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_44/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dropout_tf_45_layer_call_and_return_conditional_losses_9168772'
%dropout_tf_45/StatefulPartitionedCallµ
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_45/StatefulPartitionedCall:output:0output_916941output_916943*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9168952 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_44/StatefulPartitionedCall&^dropout_tf_45/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2N
%dropout_tf_44/StatefulPartitionedCall%dropout_tf_44/StatefulPartitionedCall2N
%dropout_tf_45/StatefulPartitionedCall%dropout_tf_45/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î)

H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_917043

inputs,
(output__1_matmul_readvariableop_resource-
)output__1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity¢output/BiasAdd/ReadVariableOp¢output/MatMul/ReadVariableOp¢ output_-1/BiasAdd/ReadVariableOp¢output_-1/MatMul/ReadVariableOp
dropout_tf_44/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ª¤?2
dropout_tf_44/dropout/Const
dropout_tf_44/dropout/MulMulinputs$dropout_tf_44/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_tf_44/dropout/Mulp
dropout_tf_44/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_tf_44/dropout/ShapeÞ
2dropout_tf_44/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_44/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype024
2dropout_tf_44/dropout/random_uniform/RandomUniform
$dropout_tf_44/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;2&
$dropout_tf_44/dropout/GreaterEqual/yö
"dropout_tf_44/dropout/GreaterEqualGreaterEqual;dropout_tf_44/dropout/random_uniform/RandomUniform:output:0-dropout_tf_44/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"dropout_tf_44/dropout/GreaterEqual©
dropout_tf_44/dropout/CastCast&dropout_tf_44/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_tf_44/dropout/Cast²
dropout_tf_44/dropout/Mul_1Muldropout_tf_44/dropout/Mul:z:0dropout_tf_44/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_tf_44/dropout/Mul_1«
output_-1/MatMul/ReadVariableOpReadVariableOp(output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
output_-1/MatMul/ReadVariableOpª
output_-1/MatMulMatMuldropout_tf_44/dropout/Mul_1:z:0'output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
output_-1/MatMulª
 output_-1/BiasAdd/ReadVariableOpReadVariableOp)output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 output_-1/BiasAdd/ReadVariableOp©
output_-1/BiasAddBiasAddoutput_-1/MatMul:product:0(output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
output_-1/BiasAddv
output_-1/ReluReluoutput_-1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
output_-1/Relu
dropout_tf_45/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ª¤?2
dropout_tf_45/dropout/Const³
dropout_tf_45/dropout/MulMuloutput_-1/Relu:activations:0$dropout_tf_45/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout_tf_45/dropout/Mul
dropout_tf_45/dropout/ShapeShapeoutput_-1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_tf_45/dropout/ShapeÞ
2dropout_tf_45/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_45/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype024
2dropout_tf_45/dropout/random_uniform/RandomUniform
$dropout_tf_45/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;2&
$dropout_tf_45/dropout/GreaterEqual/yö
"dropout_tf_45/dropout/GreaterEqualGreaterEqual;dropout_tf_45/dropout/random_uniform/RandomUniform:output:0-dropout_tf_45/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22$
"dropout_tf_45/dropout/GreaterEqual©
dropout_tf_45/dropout/CastCast&dropout_tf_45/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout_tf_45/dropout/Cast²
dropout_tf_45/dropout/Mul_1Muldropout_tf_45/dropout/Mul:z:0dropout_tf_45/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout_tf_45/dropout/Mul_1¢
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOp¡
output/MatMulMatMuldropout_tf_45/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/MatMul¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/BiasAddï
IdentityIdentityoutput/BiasAdd:output:0^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp!^output_-1/BiasAdd/ReadVariableOp ^output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2D
 output_-1/BiasAdd/ReadVariableOp output_-1/BiasAdd/ReadVariableOp2B
output_-1/MatMul/ReadVariableOpoutput_-1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Í
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_916912	
input
output__1_916860
output__1_916862
output_916906
output_916908
identity¢%dropout_tf_44/StatefulPartitionedCall¢%dropout_tf_45/StatefulPartitionedCall¢output/StatefulPartitionedCall¢!output_-1/StatefulPartitionedCallý
%dropout_tf_44/StatefulPartitionedCallStatefulPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dropout_tf_44_layer_call_and_return_conditional_losses_9168302'
%dropout_tf_44/StatefulPartitionedCallÄ
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_44/StatefulPartitionedCall:output:0output__1_916860output__1_916862*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_-1_layer_call_and_return_conditional_losses_9168492#
!output_-1/StatefulPartitionedCallÊ
%dropout_tf_45/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_44/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dropout_tf_45_layer_call_and_return_conditional_losses_9168772'
%dropout_tf_45/StatefulPartitionedCallµ
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_45/StatefulPartitionedCall:output:0output_916906output_916908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_9168952 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_44/StatefulPartitionedCall&^dropout_tf_45/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2N
%dropout_tf_44/StatefulPartitionedCall%dropout_tf_44/StatefulPartitionedCall2N
%dropout_tf_45/StatefulPartitionedCall%dropout_tf_45/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
î)

H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_917076

inputs,
(output__1_matmul_readvariableop_resource-
)output__1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity¢output/BiasAdd/ReadVariableOp¢output/MatMul/ReadVariableOp¢ output_-1/BiasAdd/ReadVariableOp¢output_-1/MatMul/ReadVariableOp
dropout_tf_44/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ª¤?2
dropout_tf_44/dropout/Const
dropout_tf_44/dropout/MulMulinputs$dropout_tf_44/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_tf_44/dropout/Mulp
dropout_tf_44/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_tf_44/dropout/ShapeÞ
2dropout_tf_44/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_44/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype024
2dropout_tf_44/dropout/random_uniform/RandomUniform
$dropout_tf_44/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;2&
$dropout_tf_44/dropout/GreaterEqual/yö
"dropout_tf_44/dropout/GreaterEqualGreaterEqual;dropout_tf_44/dropout/random_uniform/RandomUniform:output:0-dropout_tf_44/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"dropout_tf_44/dropout/GreaterEqual©
dropout_tf_44/dropout/CastCast&dropout_tf_44/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_tf_44/dropout/Cast²
dropout_tf_44/dropout/Mul_1Muldropout_tf_44/dropout/Mul:z:0dropout_tf_44/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_tf_44/dropout/Mul_1«
output_-1/MatMul/ReadVariableOpReadVariableOp(output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
output_-1/MatMul/ReadVariableOpª
output_-1/MatMulMatMuldropout_tf_44/dropout/Mul_1:z:0'output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
output_-1/MatMulª
 output_-1/BiasAdd/ReadVariableOpReadVariableOp)output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 output_-1/BiasAdd/ReadVariableOp©
output_-1/BiasAddBiasAddoutput_-1/MatMul:product:0(output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
output_-1/BiasAddv
output_-1/ReluReluoutput_-1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
output_-1/Relu
dropout_tf_45/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ª¤?2
dropout_tf_45/dropout/Const³
dropout_tf_45/dropout/MulMuloutput_-1/Relu:activations:0$dropout_tf_45/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout_tf_45/dropout/Mul
dropout_tf_45/dropout/ShapeShapeoutput_-1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_tf_45/dropout/ShapeÞ
2dropout_tf_45/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_45/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype024
2dropout_tf_45/dropout/random_uniform/RandomUniform
$dropout_tf_45/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;2&
$dropout_tf_45/dropout/GreaterEqual/yö
"dropout_tf_45/dropout/GreaterEqualGreaterEqual;dropout_tf_45/dropout/random_uniform/RandomUniform:output:0-dropout_tf_45/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22$
"dropout_tf_45/dropout/GreaterEqual©
dropout_tf_45/dropout/CastCast&dropout_tf_45/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout_tf_45/dropout/Cast²
dropout_tf_45/dropout/Mul_1Muldropout_tf_45/dropout/Mul:z:0dropout_tf_45/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout_tf_45/dropout/Mul_1¢
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOp¡
output/MatMulMatMuldropout_tf_45/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/MatMul¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/BiasAddï
IdentityIdentityoutput/BiasAdd:output:0^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp!^output_-1/BiasAdd/ReadVariableOp ^output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2D
 output_-1/BiasAdd/ReadVariableOp output_-1/BiasAdd/ReadVariableOp2B
output_-1/MatMul/ReadVariableOpoutput_-1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï	
Þ
E__inference_output_-1_layer_call_and_return_conditional_losses_916849

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û3
ß
!__inference__wrapped_model_916814	
input9
5mc_dropout_3_output__1_matmul_readvariableop_resource:
6mc_dropout_3_output__1_biasadd_readvariableop_resource6
2mc_dropout_3_output_matmul_readvariableop_resource7
3mc_dropout_3_output_biasadd_readvariableop_resource
identity¢*mc_dropout_3/output/BiasAdd/ReadVariableOp¢)mc_dropout_3/output/MatMul/ReadVariableOp¢-mc_dropout_3/output_-1/BiasAdd/ReadVariableOp¢,mc_dropout_3/output_-1/MatMul/ReadVariableOp
(mc_dropout_3/dropout_tf_44/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ª¤?2*
(mc_dropout_3/dropout_tf_44/dropout/ConstÃ
&mc_dropout_3/dropout_tf_44/dropout/MulMulinput1mc_dropout_3/dropout_tf_44/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&mc_dropout_3/dropout_tf_44/dropout/Mul
(mc_dropout_3/dropout_tf_44/dropout/ShapeShapeinput*
T0*
_output_shapes
:2*
(mc_dropout_3/dropout_tf_44/dropout/Shape
?mc_dropout_3/dropout_tf_44/dropout/random_uniform/RandomUniformRandomUniform1mc_dropout_3/dropout_tf_44/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02A
?mc_dropout_3/dropout_tf_44/dropout/random_uniform/RandomUniform«
1mc_dropout_3/dropout_tf_44/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;23
1mc_dropout_3/dropout_tf_44/dropout/GreaterEqual/yª
/mc_dropout_3/dropout_tf_44/dropout/GreaterEqualGreaterEqualHmc_dropout_3/dropout_tf_44/dropout/random_uniform/RandomUniform:output:0:mc_dropout_3/dropout_tf_44/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/mc_dropout_3/dropout_tf_44/dropout/GreaterEqualÐ
'mc_dropout_3/dropout_tf_44/dropout/CastCast3mc_dropout_3/dropout_tf_44/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'mc_dropout_3/dropout_tf_44/dropout/Castæ
(mc_dropout_3/dropout_tf_44/dropout/Mul_1Mul*mc_dropout_3/dropout_tf_44/dropout/Mul:z:0+mc_dropout_3/dropout_tf_44/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(mc_dropout_3/dropout_tf_44/dropout/Mul_1Ò
,mc_dropout_3/output_-1/MatMul/ReadVariableOpReadVariableOp5mc_dropout_3_output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02.
,mc_dropout_3/output_-1/MatMul/ReadVariableOpÞ
mc_dropout_3/output_-1/MatMulMatMul,mc_dropout_3/dropout_tf_44/dropout/Mul_1:z:04mc_dropout_3/output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
mc_dropout_3/output_-1/MatMulÑ
-mc_dropout_3/output_-1/BiasAdd/ReadVariableOpReadVariableOp6mc_dropout_3_output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02/
-mc_dropout_3/output_-1/BiasAdd/ReadVariableOpÝ
mc_dropout_3/output_-1/BiasAddBiasAdd'mc_dropout_3/output_-1/MatMul:product:05mc_dropout_3/output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
mc_dropout_3/output_-1/BiasAdd
mc_dropout_3/output_-1/ReluRelu'mc_dropout_3/output_-1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
mc_dropout_3/output_-1/Relu
(mc_dropout_3/dropout_tf_45/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ª¤?2*
(mc_dropout_3/dropout_tf_45/dropout/Constç
&mc_dropout_3/dropout_tf_45/dropout/MulMul)mc_dropout_3/output_-1/Relu:activations:01mc_dropout_3/dropout_tf_45/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22(
&mc_dropout_3/dropout_tf_45/dropout/Mul­
(mc_dropout_3/dropout_tf_45/dropout/ShapeShape)mc_dropout_3/output_-1/Relu:activations:0*
T0*
_output_shapes
:2*
(mc_dropout_3/dropout_tf_45/dropout/Shape
?mc_dropout_3/dropout_tf_45/dropout/random_uniform/RandomUniformRandomUniform1mc_dropout_3/dropout_tf_45/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype02A
?mc_dropout_3/dropout_tf_45/dropout/random_uniform/RandomUniform«
1mc_dropout_3/dropout_tf_45/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;23
1mc_dropout_3/dropout_tf_45/dropout/GreaterEqual/yª
/mc_dropout_3/dropout_tf_45/dropout/GreaterEqualGreaterEqualHmc_dropout_3/dropout_tf_45/dropout/random_uniform/RandomUniform:output:0:mc_dropout_3/dropout_tf_45/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ221
/mc_dropout_3/dropout_tf_45/dropout/GreaterEqualÐ
'mc_dropout_3/dropout_tf_45/dropout/CastCast3mc_dropout_3/dropout_tf_45/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22)
'mc_dropout_3/dropout_tf_45/dropout/Castæ
(mc_dropout_3/dropout_tf_45/dropout/Mul_1Mul*mc_dropout_3/dropout_tf_45/dropout/Mul:z:0+mc_dropout_3/dropout_tf_45/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
(mc_dropout_3/dropout_tf_45/dropout/Mul_1É
)mc_dropout_3/output/MatMul/ReadVariableOpReadVariableOp2mc_dropout_3_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)mc_dropout_3/output/MatMul/ReadVariableOpÕ
mc_dropout_3/output/MatMulMatMul,mc_dropout_3/dropout_tf_45/dropout/Mul_1:z:01mc_dropout_3/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mc_dropout_3/output/MatMulÈ
*mc_dropout_3/output/BiasAdd/ReadVariableOpReadVariableOp3mc_dropout_3_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*mc_dropout_3/output/BiasAdd/ReadVariableOpÑ
mc_dropout_3/output/BiasAddBiasAdd$mc_dropout_3/output/MatMul:product:02mc_dropout_3/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mc_dropout_3/output/BiasAdd°
IdentityIdentity$mc_dropout_3/output/BiasAdd:output:0+^mc_dropout_3/output/BiasAdd/ReadVariableOp*^mc_dropout_3/output/MatMul/ReadVariableOp.^mc_dropout_3/output_-1/BiasAdd/ReadVariableOp-^mc_dropout_3/output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2X
*mc_dropout_3/output/BiasAdd/ReadVariableOp*mc_dropout_3/output/BiasAdd/ReadVariableOp2V
)mc_dropout_3/output/MatMul/ReadVariableOp)mc_dropout_3/output/MatMul/ReadVariableOp2^
-mc_dropout_3/output_-1/BiasAdd/ReadVariableOp-mc_dropout_3/output_-1/BiasAdd/ReadVariableOp2\
,mc_dropout_3/output_-1/MatMul/ReadVariableOp,mc_dropout_3/output_-1/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
á

*__inference_output_-1_layer_call_fn_917139

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_-1_layer_call_and_return_conditional_losses_9168492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
g
.__inference_dropout_tf_45_layer_call_fn_917156

inputs
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dropout_tf_45_layer_call_and_return_conditional_losses_9168772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ222
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs

h
I__inference_dropout_tf_45_layer_call_and_return_conditional_losses_916877

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ª¤?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
×£;2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¥
serving_default
7
input.
serving_default_input:0ÿÿÿÿÿÿÿÿÿ:
output0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ò
º'
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
^_default_save_signature"Ö$
_tf_keras_networkº${"class_name": "MCDropout", "name": "mc_dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "mc_dropout_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_44", "trainable": true, "dtype": "float32", "rate": 0.005, "noise_shape": null, "seed": null}, "name": "dropout_tf_44", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_-1", "inbound_nodes": [[["dropout_tf_44", 0, 0, {}]]]}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_45", "trainable": true, "dtype": "float32", "rate": 0.005, "noise_shape": null, "seed": null}, "name": "dropout_tf_45", "inbound_nodes": [[["output_-1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_tf_45", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MCDropout", "config": {"name": "mc_dropout_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_44", "trainable": true, "dtype": "float32", "rate": 0.005, "noise_shape": null, "seed": null}, "name": "dropout_tf_44", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_-1", "inbound_nodes": [[["dropout_tf_44", 0, 0, {}]]]}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_45", "trainable": true, "dtype": "float32", "rate": 0.005, "noise_shape": null, "seed": null}, "name": "dropout_tf_45", "inbound_nodes": [[["output_-1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_tf_45", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "nll", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
å"â
_tf_keras_input_layerÂ{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
ò
trainable_variables
regularization_losses
	variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"ã
_tf_keras_layerÉ{"class_name": "DropoutTF", "name": "dropout_tf_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_tf_44", "trainable": true, "dtype": "float32", "rate": 0.005, "noise_shape": null, "seed": null}}
ò

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "output_-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
ò
trainable_variables
regularization_losses
	variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"ã
_tf_keras_layerÉ{"class_name": "DropoutTF", "name": "dropout_tf_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_tf_45", "trainable": true, "dtype": "float32", "rate": 0.005, "noise_shape": null, "seed": null}}
ï

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
*e&call_and_return_all_conditional_losses
f__call__"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
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
Ê
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
": 22output_-1/kernel
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
»
	Ftotal
	Gcount
H	variables
I	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ô
	Jtotal
	Kcount
L
_fn_kwargs
M	variables
N	keras_api"­
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
ó
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api"¬
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
':%22Adam/output_-1/kernel/m
!:22Adam/output_-1/bias/m
$:"22Adam/output/kernel/m
:2Adam/output/bias/m
':%22Adam/output_-1/kernel/v
!:22Adam/output_-1/bias/v
$:"22Adam/output/kernel/v
:2Adam/output/bias/v
î2ë
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_916928
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_917076
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_917043
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_916912À
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
2ÿ
-__inference_mc_dropout_3_layer_call_fn_916987
-__inference_mc_dropout_3_layer_call_fn_916958
-__inference_mc_dropout_3_layer_call_fn_917089
-__inference_mc_dropout_3_layer_call_fn_917102À
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
Ý2Ú
!__inference__wrapped_model_916814´
²
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
annotationsª *$¢!

inputÿÿÿÿÿÿÿÿÿ
ó2ð
I__inference_dropout_tf_44_layer_call_and_return_conditional_losses_917114¢
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
Ø2Õ
.__inference_dropout_tf_44_layer_call_fn_917119¢
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
ï2ì
E__inference_output_-1_layer_call_and_return_conditional_losses_917130¢
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
Ô2Ñ
*__inference_output_-1_layer_call_fn_917139¢
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
I__inference_dropout_tf_45_layer_call_and_return_conditional_losses_917151¢
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
Ø2Õ
.__inference_dropout_tf_45_layer_call_fn_917156¢
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
ì2é
B__inference_output_layer_call_and_return_conditional_losses_917166¢
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
Ñ2Î
'__inference_output_layer_call_fn_917175¢
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
ÉBÆ
$__inference_signature_wrapper_917010input"
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
 
!__inference__wrapped_model_916814g.¢+
$¢!

inputÿÿÿÿÿÿÿÿÿ
ª "/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿ¥
I__inference_dropout_tf_44_layer_call_and_return_conditional_losses_917114X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
.__inference_dropout_tf_44_layer_call_fn_917119K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
I__inference_dropout_tf_45_layer_call_and_return_conditional_losses_917151X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 }
.__inference_dropout_tf_45_layer_call_fn_917156K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿ2±
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_916912e6¢3
,¢)

inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_916928e6¢3
,¢)

inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ²
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_917043f7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ²
H__inference_mc_dropout_3_layer_call_and_return_conditional_losses_917076f7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_mc_dropout_3_layer_call_fn_916958X6¢3
,¢)

inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_mc_dropout_3_layer_call_fn_916987X6¢3
,¢)

inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_mc_dropout_3_layer_call_fn_917089Y7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_mc_dropout_3_layer_call_fn_917102Y7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_output_-1_layer_call_and_return_conditional_losses_917130\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ2
 }
*__inference_output_-1_layer_call_fn_917139O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ2¢
B__inference_output_layer_call_and_return_conditional_losses_917166\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_output_layer_call_fn_917175O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ2
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_signature_wrapper_917010p7¢4
¢ 
-ª*
(
input
inputÿÿÿÿÿÿÿÿÿ"/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿ