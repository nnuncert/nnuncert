ий
╘к
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
╛
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
executor_typestring И
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-0-ga4dfb8d1a718╪ї
|
output_-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*!
shared_nameoutput_-1/kernel
u
$output_-1/kernel/Read/ReadVariableOpReadVariableOpoutput_-1/kernel*
_output_shapes

:2*
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
К
Adam/output_-1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/output_-1/kernel/m
Г
+Adam/output_-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_-1/kernel/m*
_output_shapes

:2*
dtype0
В
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
Д
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
К
Adam/output_-1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*(
shared_nameAdam/output_-1/kernel/v
Г
+Adam/output_-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_-1/kernel/v*
_output_shapes

:2*
dtype0
В
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
Д
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
У$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╬#
value─#B┴# B║#
К
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
		variables

regularization_losses
trainable_variables
	keras_api

signatures
 
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api

"grid_dropout

#1
И
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
н
)non_trainable_variables

*layers
+layer_regularization_losses
		variables

regularization_losses
,metrics
trainable_variables
-layer_metrics
 
 
 
 
н
.non_trainable_variables

/layers
0layer_regularization_losses
	variables
regularization_losses
1metrics
trainable_variables
2layer_metrics
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
н
3non_trainable_variables

4layers
5layer_regularization_losses
	variables
regularization_losses
6metrics
trainable_variables
7layer_metrics
 
 
 
н
8non_trainable_variables

9layers
:layer_regularization_losses
	variables
regularization_losses
;metrics
trainable_variables
<layer_metrics
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
н
=non_trainable_variables

>layers
?layer_regularization_losses
	variables
regularization_losses
@metrics
 trainable_variables
Alayer_metrics
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
#
0
1
2
3
4
 

C0
D1
E2
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
:         *
dtype0*
shape:         
·
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputoutput_-1/kerneloutput_-1/biasoutput/kerneloutput/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_2432616
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
GPU2*0J 8В *)
f$R"
 __inference__traced_save_2432873
Ъ
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
GPU2*0J 8В *,
f'R%
#__inference__traced_restore_2432952 Х
№
╤
.__inference_mc_dropout_8_layer_call_fn_2432642

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_24325392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
о

ў
F__inference_output_-1_layer_call_and_return_conditional_losses_2432428

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         22
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
░
i
J__inference_dropout_tf_69_layer_call_and_return_conditional_losses_2432762

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
в
Ш
+__inference_output_-1_layer_call_fn_2432734

inputs
unknown:2
	unknown_0:2
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_output_-1_layer_call_and_return_conditional_losses_24324282
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╔
╟
%__inference_signature_wrapper_2432616	
input
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:
identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_24323962
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:         

_user_specified_nameinput
ёb
П
#__inference__traced_restore_2432952
file_prefix3
!assignvariableop_output__1_kernel:2/
!assignvariableop_1_output__1_bias:22
 assignvariableop_2_output_kernel:2,
assignvariableop_3_output_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: %
assignvariableop_13_total_2: %
assignvariableop_14_count_2: =
+assignvariableop_15_adam_output__1_kernel_m:27
)assignvariableop_16_adam_output__1_bias_m:2:
(assignvariableop_17_adam_output_kernel_m:24
&assignvariableop_18_adam_output_bias_m:=
+assignvariableop_19_adam_output__1_kernel_v:27
)assignvariableop_20_adam_output__1_bias_v:2:
(assignvariableop_21_adam_output_kernel_v:24
&assignvariableop_22_adam_output_bias_v:
identity_24ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9─
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╨
value╞B├B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╛
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesг
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

Identityа
AssignVariableOpAssignVariableOp!assignvariableop_output__1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ж
AssignVariableOp_1AssignVariableOp!assignvariableop_1_output__1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2е
AssignVariableOp_2AssignVariableOp assignvariableop_2_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3г
AssignVariableOp_3AssignVariableOpassignvariableop_3_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4б
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5г
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6г
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7в
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8к
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Э
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10б
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11г
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12г
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13г
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14г
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15│
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_output__1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16▒
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_output__1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17░
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_output_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18о
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_output_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19│
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_output__1_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20▒
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_output__1_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21░
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_output_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22о
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_output_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_229
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╪
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23╦
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2$
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
о

ў
F__inference_output_-1_layer_call_and_return_conditional_losses_2432745

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         22
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
т
Г
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432539

inputs#
output__1_2432527:2
output__1_2432529:2 
output_2432533:2
output_2432535:
identityИв%dropout_tf_68/StatefulPartitionedCallв%dropout_tf_69/StatefulPartitionedCallвoutput/StatefulPartitionedCallв!output_-1/StatefulPartitionedCall 
%dropout_tf_68/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dropout_tf_68_layer_call_and_return_conditional_losses_24324152'
%dropout_tf_68/StatefulPartitionedCall╟
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_68/StatefulPartitionedCall:output:0output__1_2432527output__1_2432529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_output_-1_layer_call_and_return_conditional_losses_24324282#
!output_-1/StatefulPartitionedCall╦
%dropout_tf_69/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_68/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dropout_tf_69_layer_call_and_return_conditional_losses_24324462'
%dropout_tf_69/StatefulPartitionedCall╕
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_69/StatefulPartitionedCall:output:0output_2432533output_2432535*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_24324582 
output/StatefulPartitionedCallР
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_68/StatefulPartitionedCall&^dropout_tf_69/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2N
%dropout_tf_68/StatefulPartitionedCall%dropout_tf_68/StatefulPartitionedCall2N
%dropout_tf_69/StatefulPartitionedCall%dropout_tf_69/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┴*
╨
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432708

inputs:
(output__1_matmul_readvariableop_resource:27
)output__1_biasadd_readvariableop_resource:27
%output_matmul_readvariableop_resource:24
&output_biasadd_readvariableop_resource:
identityИвoutput/BiasAdd/ReadVariableOpвoutput/MatMul/ReadVariableOpв output_-1/BiasAdd/ReadVariableOpвoutput_-1/MatMul/ReadVariableOp
dropout_tf_68/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?2
dropout_tf_68/dropout/ConstЭ
dropout_tf_68/dropout/MulMulinputs$dropout_tf_68/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_tf_68/dropout/Mulp
dropout_tf_68/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_tf_68/dropout/Shape▐
2dropout_tf_68/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_68/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype024
2dropout_tf_68/dropout/random_uniform/RandomUniformС
$dropout_tf_68/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2&
$dropout_tf_68/dropout/GreaterEqual/yЎ
"dropout_tf_68/dropout/GreaterEqualGreaterEqual;dropout_tf_68/dropout/random_uniform/RandomUniform:output:0-dropout_tf_68/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2$
"dropout_tf_68/dropout/GreaterEqualй
dropout_tf_68/dropout/CastCast&dropout_tf_68/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_tf_68/dropout/Cast▓
dropout_tf_68/dropout/Mul_1Muldropout_tf_68/dropout/Mul:z:0dropout_tf_68/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_tf_68/dropout/Mul_1л
output_-1/MatMul/ReadVariableOpReadVariableOp(output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
output_-1/MatMul/ReadVariableOpк
output_-1/MatMulMatMuldropout_tf_68/dropout/Mul_1:z:0'output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
output_-1/MatMulк
 output_-1/BiasAdd/ReadVariableOpReadVariableOp)output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 output_-1/BiasAdd/ReadVariableOpй
output_-1/BiasAddBiasAddoutput_-1/MatMul:product:0(output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
output_-1/BiasAddv
output_-1/ReluReluoutput_-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
output_-1/Relu
dropout_tf_69/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?2
dropout_tf_69/dropout/Const│
dropout_tf_69/dropout/MulMuloutput_-1/Relu:activations:0$dropout_tf_69/dropout/Const:output:0*
T0*'
_output_shapes
:         22
dropout_tf_69/dropout/MulЖ
dropout_tf_69/dropout/ShapeShapeoutput_-1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_tf_69/dropout/Shape▐
2dropout_tf_69/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_69/dropout/Shape:output:0*
T0*'
_output_shapes
:         2*
dtype024
2dropout_tf_69/dropout/random_uniform/RandomUniformС
$dropout_tf_69/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2&
$dropout_tf_69/dropout/GreaterEqual/yЎ
"dropout_tf_69/dropout/GreaterEqualGreaterEqual;dropout_tf_69/dropout/random_uniform/RandomUniform:output:0-dropout_tf_69/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         22$
"dropout_tf_69/dropout/GreaterEqualй
dropout_tf_69/dropout/CastCast&dropout_tf_69/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         22
dropout_tf_69/dropout/Cast▓
dropout_tf_69/dropout/Mul_1Muldropout_tf_69/dropout/Mul:z:0dropout_tf_69/dropout/Cast:y:0*
T0*'
_output_shapes
:         22
dropout_tf_69/dropout/Mul_1в
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOpб
output/MatMulMatMuldropout_tf_69/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulб
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAddя
IdentityIdentityoutput/BiasAdd:output:0^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp!^output_-1/BiasAdd/ReadVariableOp ^output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2D
 output_-1/BiasAdd/ReadVariableOp output_-1/BiasAdd/ReadVariableOp2B
output_-1/MatMul/ReadVariableOpoutput_-1/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
о4
Р
"__inference__wrapped_model_2432396	
inputG
5mc_dropout_8_output__1_matmul_readvariableop_resource:2D
6mc_dropout_8_output__1_biasadd_readvariableop_resource:2D
2mc_dropout_8_output_matmul_readvariableop_resource:2A
3mc_dropout_8_output_biasadd_readvariableop_resource:
identityИв*mc_dropout_8/output/BiasAdd/ReadVariableOpв)mc_dropout_8/output/MatMul/ReadVariableOpв-mc_dropout_8/output_-1/BiasAdd/ReadVariableOpв,mc_dropout_8/output_-1/MatMul/ReadVariableOpЩ
(mc_dropout_8/dropout_tf_68/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?2*
(mc_dropout_8/dropout_tf_68/dropout/Const├
&mc_dropout_8/dropout_tf_68/dropout/MulMulinput1mc_dropout_8/dropout_tf_68/dropout/Const:output:0*
T0*'
_output_shapes
:         2(
&mc_dropout_8/dropout_tf_68/dropout/MulЙ
(mc_dropout_8/dropout_tf_68/dropout/ShapeShapeinput*
T0*
_output_shapes
:2*
(mc_dropout_8/dropout_tf_68/dropout/ShapeЕ
?mc_dropout_8/dropout_tf_68/dropout/random_uniform/RandomUniformRandomUniform1mc_dropout_8/dropout_tf_68/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02A
?mc_dropout_8/dropout_tf_68/dropout/random_uniform/RandomUniformл
1mc_dropout_8/dropout_tf_68/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<23
1mc_dropout_8/dropout_tf_68/dropout/GreaterEqual/yк
/mc_dropout_8/dropout_tf_68/dropout/GreaterEqualGreaterEqualHmc_dropout_8/dropout_tf_68/dropout/random_uniform/RandomUniform:output:0:mc_dropout_8/dropout_tf_68/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         21
/mc_dropout_8/dropout_tf_68/dropout/GreaterEqual╨
'mc_dropout_8/dropout_tf_68/dropout/CastCast3mc_dropout_8/dropout_tf_68/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2)
'mc_dropout_8/dropout_tf_68/dropout/Castц
(mc_dropout_8/dropout_tf_68/dropout/Mul_1Mul*mc_dropout_8/dropout_tf_68/dropout/Mul:z:0+mc_dropout_8/dropout_tf_68/dropout/Cast:y:0*
T0*'
_output_shapes
:         2*
(mc_dropout_8/dropout_tf_68/dropout/Mul_1╥
,mc_dropout_8/output_-1/MatMul/ReadVariableOpReadVariableOp5mc_dropout_8_output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02.
,mc_dropout_8/output_-1/MatMul/ReadVariableOp▐
mc_dropout_8/output_-1/MatMulMatMul,mc_dropout_8/dropout_tf_68/dropout/Mul_1:z:04mc_dropout_8/output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
mc_dropout_8/output_-1/MatMul╤
-mc_dropout_8/output_-1/BiasAdd/ReadVariableOpReadVariableOp6mc_dropout_8_output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02/
-mc_dropout_8/output_-1/BiasAdd/ReadVariableOp▌
mc_dropout_8/output_-1/BiasAddBiasAdd'mc_dropout_8/output_-1/MatMul:product:05mc_dropout_8/output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22 
mc_dropout_8/output_-1/BiasAddЭ
mc_dropout_8/output_-1/ReluRelu'mc_dropout_8/output_-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
mc_dropout_8/output_-1/ReluЩ
(mc_dropout_8/dropout_tf_69/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?2*
(mc_dropout_8/dropout_tf_69/dropout/Constч
&mc_dropout_8/dropout_tf_69/dropout/MulMul)mc_dropout_8/output_-1/Relu:activations:01mc_dropout_8/dropout_tf_69/dropout/Const:output:0*
T0*'
_output_shapes
:         22(
&mc_dropout_8/dropout_tf_69/dropout/Mulн
(mc_dropout_8/dropout_tf_69/dropout/ShapeShape)mc_dropout_8/output_-1/Relu:activations:0*
T0*
_output_shapes
:2*
(mc_dropout_8/dropout_tf_69/dropout/ShapeЕ
?mc_dropout_8/dropout_tf_69/dropout/random_uniform/RandomUniformRandomUniform1mc_dropout_8/dropout_tf_69/dropout/Shape:output:0*
T0*'
_output_shapes
:         2*
dtype02A
?mc_dropout_8/dropout_tf_69/dropout/random_uniform/RandomUniformл
1mc_dropout_8/dropout_tf_69/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<23
1mc_dropout_8/dropout_tf_69/dropout/GreaterEqual/yк
/mc_dropout_8/dropout_tf_69/dropout/GreaterEqualGreaterEqualHmc_dropout_8/dropout_tf_69/dropout/random_uniform/RandomUniform:output:0:mc_dropout_8/dropout_tf_69/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         221
/mc_dropout_8/dropout_tf_69/dropout/GreaterEqual╨
'mc_dropout_8/dropout_tf_69/dropout/CastCast3mc_dropout_8/dropout_tf_69/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         22)
'mc_dropout_8/dropout_tf_69/dropout/Castц
(mc_dropout_8/dropout_tf_69/dropout/Mul_1Mul*mc_dropout_8/dropout_tf_69/dropout/Mul:z:0+mc_dropout_8/dropout_tf_69/dropout/Cast:y:0*
T0*'
_output_shapes
:         22*
(mc_dropout_8/dropout_tf_69/dropout/Mul_1╔
)mc_dropout_8/output/MatMul/ReadVariableOpReadVariableOp2mc_dropout_8_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02+
)mc_dropout_8/output/MatMul/ReadVariableOp╒
mc_dropout_8/output/MatMulMatMul,mc_dropout_8/dropout_tf_69/dropout/Mul_1:z:01mc_dropout_8/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
mc_dropout_8/output/MatMul╚
*mc_dropout_8/output/BiasAdd/ReadVariableOpReadVariableOp3mc_dropout_8_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*mc_dropout_8/output/BiasAdd/ReadVariableOp╤
mc_dropout_8/output/BiasAddBiasAdd$mc_dropout_8/output/MatMul:product:02mc_dropout_8/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
mc_dropout_8/output/BiasAdd░
IdentityIdentity$mc_dropout_8/output/BiasAdd:output:0+^mc_dropout_8/output/BiasAdd/ReadVariableOp*^mc_dropout_8/output/MatMul/ReadVariableOp.^mc_dropout_8/output_-1/BiasAdd/ReadVariableOp-^mc_dropout_8/output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2X
*mc_dropout_8/output/BiasAdd/ReadVariableOp*mc_dropout_8/output/BiasAdd/ReadVariableOp2V
)mc_dropout_8/output/MatMul/ReadVariableOp)mc_dropout_8/output/MatMul/ReadVariableOp2^
-mc_dropout_8/output_-1/BiasAdd/ReadVariableOp-mc_dropout_8/output_-1/BiasAdd/ReadVariableOp2\
,mc_dropout_8/output_-1/MatMul/ReadVariableOp,mc_dropout_8/output_-1/MatMul/ReadVariableOp:N J
'
_output_shapes
:         

_user_specified_nameinput
∙
╨
.__inference_mc_dropout_8_layer_call_fn_2432476	
input
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:
identityИвStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_24324652
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:         

_user_specified_nameinput
▀
В
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432595	
input#
output__1_2432583:2
output__1_2432585:2 
output_2432589:2
output_2432591:
identityИв%dropout_tf_68/StatefulPartitionedCallв%dropout_tf_69/StatefulPartitionedCallвoutput/StatefulPartitionedCallв!output_-1/StatefulPartitionedCall■
%dropout_tf_68/StatefulPartitionedCallStatefulPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dropout_tf_68_layer_call_and_return_conditional_losses_24324152'
%dropout_tf_68/StatefulPartitionedCall╟
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_68/StatefulPartitionedCall:output:0output__1_2432583output__1_2432585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_output_-1_layer_call_and_return_conditional_losses_24324282#
!output_-1/StatefulPartitionedCall╦
%dropout_tf_69/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_68/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dropout_tf_69_layer_call_and_return_conditional_losses_24324462'
%dropout_tf_69/StatefulPartitionedCall╕
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_69/StatefulPartitionedCall:output:0output_2432589output_2432591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_24324582 
output/StatefulPartitionedCallР
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_68/StatefulPartitionedCall&^dropout_tf_69/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2N
%dropout_tf_68/StatefulPartitionedCall%dropout_tf_68/StatefulPartitionedCall2N
%dropout_tf_69/StatefulPartitionedCall%dropout_tf_69/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:N J
'
_output_shapes
:         

_user_specified_nameinput
№
╤
.__inference_mc_dropout_8_layer_call_fn_2432629

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_24324652
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
░
i
J__inference_dropout_tf_68_layer_call_and_return_conditional_losses_2432725

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╧	
Ї
C__inference_output_layer_call_and_return_conditional_losses_2432781

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
╧	
Ї
C__inference_output_layer_call_and_return_conditional_losses_2432458

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
░
i
J__inference_dropout_tf_69_layer_call_and_return_conditional_losses_2432446

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
░
i
J__inference_dropout_tf_68_layer_call_and_return_conditional_losses_2432415

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ь
Х
(__inference_output_layer_call_fn_2432771

inputs
unknown:2
	unknown_0:
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_24324582
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
т
Г
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432465

inputs#
output__1_2432429:2
output__1_2432431:2 
output_2432459:2
output_2432461:
identityИв%dropout_tf_68/StatefulPartitionedCallв%dropout_tf_69/StatefulPartitionedCallвoutput/StatefulPartitionedCallв!output_-1/StatefulPartitionedCall 
%dropout_tf_68/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dropout_tf_68_layer_call_and_return_conditional_losses_24324152'
%dropout_tf_68/StatefulPartitionedCall╟
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_68/StatefulPartitionedCall:output:0output__1_2432429output__1_2432431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_output_-1_layer_call_and_return_conditional_losses_24324282#
!output_-1/StatefulPartitionedCall╦
%dropout_tf_69/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_68/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dropout_tf_69_layer_call_and_return_conditional_losses_24324462'
%dropout_tf_69/StatefulPartitionedCall╕
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_69/StatefulPartitionedCall:output:0output_2432459output_2432461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_24324582 
output/StatefulPartitionedCallР
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_68/StatefulPartitionedCall&^dropout_tf_69/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2N
%dropout_tf_68/StatefulPartitionedCall%dropout_tf_68/StatefulPartitionedCall2N
%dropout_tf_69/StatefulPartitionedCall%dropout_tf_69/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
∙
╨
.__inference_mc_dropout_8_layer_call_fn_2432563	
input
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:
identityИвStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_24325392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:         

_user_specified_nameinput
╫
h
/__inference_dropout_tf_68_layer_call_fn_2432713

inputs
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dropout_tf_68_layer_call_and_return_conditional_losses_24324152
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╫
h
/__inference_dropout_tf_69_layer_call_fn_2432750

inputs
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dropout_tf_69_layer_call_and_return_conditional_losses_24324462
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         222
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
╡5
Я	
 __inference__traced_save_2432873
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

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╛
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╨
value╞B├B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╕
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesз	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_output__1_kernel_read_readvariableop)savev2_output__1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_output__1_kernel_m_read_readvariableop0savev2_adam_output__1_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_output__1_kernel_v_read_readvariableop0savev2_adam_output__1_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*Н
_input_shapes|
z: :2:2:2:: : : : : : : : : : : :2:2:2::2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:2: 
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

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:2: 
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
▀
В
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432579	
input#
output__1_2432567:2
output__1_2432569:2 
output_2432573:2
output_2432575:
identityИв%dropout_tf_68/StatefulPartitionedCallв%dropout_tf_69/StatefulPartitionedCallвoutput/StatefulPartitionedCallв!output_-1/StatefulPartitionedCall■
%dropout_tf_68/StatefulPartitionedCallStatefulPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dropout_tf_68_layer_call_and_return_conditional_losses_24324152'
%dropout_tf_68/StatefulPartitionedCall╟
!output_-1/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_68/StatefulPartitionedCall:output:0output__1_2432567output__1_2432569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_output_-1_layer_call_and_return_conditional_losses_24324282#
!output_-1/StatefulPartitionedCall╦
%dropout_tf_69/StatefulPartitionedCallStatefulPartitionedCall*output_-1/StatefulPartitionedCall:output:0&^dropout_tf_68/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dropout_tf_69_layer_call_and_return_conditional_losses_24324462'
%dropout_tf_69/StatefulPartitionedCall╕
output/StatefulPartitionedCallStatefulPartitionedCall.dropout_tf_69/StatefulPartitionedCall:output:0output_2432573output_2432575*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_24324582 
output/StatefulPartitionedCallР
IdentityIdentity'output/StatefulPartitionedCall:output:0&^dropout_tf_68/StatefulPartitionedCall&^dropout_tf_69/StatefulPartitionedCall^output/StatefulPartitionedCall"^output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2N
%dropout_tf_68/StatefulPartitionedCall%dropout_tf_68/StatefulPartitionedCall2N
%dropout_tf_69/StatefulPartitionedCall%dropout_tf_69/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2F
!output_-1/StatefulPartitionedCall!output_-1/StatefulPartitionedCall:N J
'
_output_shapes
:         

_user_specified_nameinput
┴*
╨
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432675

inputs:
(output__1_matmul_readvariableop_resource:27
)output__1_biasadd_readvariableop_resource:27
%output_matmul_readvariableop_resource:24
&output_biasadd_readvariableop_resource:
identityИвoutput/BiasAdd/ReadVariableOpвoutput/MatMul/ReadVariableOpв output_-1/BiasAdd/ReadVariableOpвoutput_-1/MatMul/ReadVariableOp
dropout_tf_68/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?2
dropout_tf_68/dropout/ConstЭ
dropout_tf_68/dropout/MulMulinputs$dropout_tf_68/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_tf_68/dropout/Mulp
dropout_tf_68/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_tf_68/dropout/Shape▐
2dropout_tf_68/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_68/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype024
2dropout_tf_68/dropout/random_uniform/RandomUniformС
$dropout_tf_68/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2&
$dropout_tf_68/dropout/GreaterEqual/yЎ
"dropout_tf_68/dropout/GreaterEqualGreaterEqual;dropout_tf_68/dropout/random_uniform/RandomUniform:output:0-dropout_tf_68/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2$
"dropout_tf_68/dropout/GreaterEqualй
dropout_tf_68/dropout/CastCast&dropout_tf_68/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_tf_68/dropout/Cast▓
dropout_tf_68/dropout/Mul_1Muldropout_tf_68/dropout/Mul:z:0dropout_tf_68/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_tf_68/dropout/Mul_1л
output_-1/MatMul/ReadVariableOpReadVariableOp(output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
output_-1/MatMul/ReadVariableOpк
output_-1/MatMulMatMuldropout_tf_68/dropout/Mul_1:z:0'output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
output_-1/MatMulк
 output_-1/BiasAdd/ReadVariableOpReadVariableOp)output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 output_-1/BiasAdd/ReadVariableOpй
output_-1/BiasAddBiasAddoutput_-1/MatMul:product:0(output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
output_-1/BiasAddv
output_-1/ReluReluoutput_-1/BiasAdd:output:0*
T0*'
_output_shapes
:         22
output_-1/Relu
dropout_tf_69/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?2
dropout_tf_69/dropout/Const│
dropout_tf_69/dropout/MulMuloutput_-1/Relu:activations:0$dropout_tf_69/dropout/Const:output:0*
T0*'
_output_shapes
:         22
dropout_tf_69/dropout/MulЖ
dropout_tf_69/dropout/ShapeShapeoutput_-1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_tf_69/dropout/Shape▐
2dropout_tf_69/dropout/random_uniform/RandomUniformRandomUniform$dropout_tf_69/dropout/Shape:output:0*
T0*'
_output_shapes
:         2*
dtype024
2dropout_tf_69/dropout/random_uniform/RandomUniformС
$dropout_tf_69/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2&
$dropout_tf_69/dropout/GreaterEqual/yЎ
"dropout_tf_69/dropout/GreaterEqualGreaterEqual;dropout_tf_69/dropout/random_uniform/RandomUniform:output:0-dropout_tf_69/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         22$
"dropout_tf_69/dropout/GreaterEqualй
dropout_tf_69/dropout/CastCast&dropout_tf_69/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         22
dropout_tf_69/dropout/Cast▓
dropout_tf_69/dropout/Mul_1Muldropout_tf_69/dropout/Mul:z:0dropout_tf_69/dropout/Cast:y:0*
T0*'
_output_shapes
:         22
dropout_tf_69/dropout/Mul_1в
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
output/MatMul/ReadVariableOpб
output/MatMulMatMuldropout_tf_69/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulб
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAddя
IdentityIdentityoutput/BiasAdd:output:0^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp!^output_-1/BiasAdd/ReadVariableOp ^output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp2D
 output_-1/BiasAdd/ReadVariableOp output_-1/BiasAdd/ReadVariableOp2B
output_-1/MatMul/ReadVariableOpoutput_-1/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*е
serving_defaultС
7
input.
serving_default_input:0         :
output0
StatefulPartitionedCall:0         tensorflow/serving/predict:Ъа
я*
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
		variables

regularization_losses
trainable_variables
	keras_api

signatures
\__call__
]_default_save_signature
*^&call_and_return_all_conditional_losses"Л(
_tf_keras_networkя'{"name": "mc_dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "MCDropout", "config": {"name": "mc_dropout_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_68", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_tf_68", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_-1", "inbound_nodes": [[["dropout_tf_68", 0, 0, {}]]]}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_69", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_tf_69", "inbound_nodes": [[["output_-1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_tf_69", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 9, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 14]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 14]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 14]}, "float32", "input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "MCDropout", "config": {"name": "mc_dropout_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_68", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_tf_68", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Dense", "config": {"name": "output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_-1", "inbound_nodes": [[["dropout_tf_68", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "DropoutTF", "config": {"name": "dropout_tf_69", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_tf_69", "inbound_nodes": [[["output_-1", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dropout_tf_69", 0, 0, {}]]], "shared_object_id": 8}], "input_layers": [["input", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": "nll", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 11}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 12}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ч"ф
_tf_keras_input_layer─{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 14]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 14]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
▓
	variables
regularization_losses
trainable_variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"г
_tf_keras_layerЙ{"name": "dropout_tf_68", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "DropoutTF", "config": {"name": "dropout_tf_68", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 1}
Г	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"▐
_tf_keras_layer─{"name": "output_-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_tf_68", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 14}}, "shared_object_id": 13}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14]}}
╢
	variables
regularization_losses
trainable_variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"з
_tf_keras_layerН{"name": "dropout_tf_69", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "DropoutTF", "config": {"name": "dropout_tf_69", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "inbound_nodes": [[["output_-1", 0, 0, {}]]], "shared_object_id": 5}
■

kernel
bias
	variables
regularization_losses
 trainable_variables
!	keras_api
e__call__
*f&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_tf_69", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
2
"grid_dropout"
trackable_dict_wrapper
(
#1"
trackable_tuple_wrapper
Ы
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
╩
)non_trainable_variables

*layers
+layer_regularization_losses
		variables

regularization_losses
,metrics
trainable_variables
-layer_metrics
\__call__
]_default_save_signature
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
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
н
.non_trainable_variables

/layers
0layer_regularization_losses
	variables
regularization_losses
1metrics
trainable_variables
2layer_metrics
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
": 22output_-1/kernel
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
н
3non_trainable_variables

4layers
5layer_regularization_losses
	variables
regularization_losses
6metrics
trainable_variables
7layer_metrics
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
8non_trainable_variables

9layers
:layer_regularization_losses
	variables
regularization_losses
;metrics
trainable_variables
<layer_metrics
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
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
н
=non_trainable_variables

>layers
?layer_regularization_losses
	variables
regularization_losses
@metrics
 trainable_variables
Alayer_metrics
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
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
C
0
1
2
3
4"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
╘
	Ftotal
	Gcount
H	variables
I	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 15}
М
	Jtotal
	Kcount
L
_fn_kwargs
M	variables
N	keras_api"┼
_tf_keras_metricк{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 11}
Л
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api"─
_tf_keras_metricй{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 12}
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
':%22Adam/output_-1/kernel/m
!:22Adam/output_-1/bias/m
$:"22Adam/output/kernel/m
:2Adam/output/bias/m
':%22Adam/output_-1/kernel/v
!:22Adam/output_-1/bias/v
$:"22Adam/output/kernel/v
:2Adam/output/bias/v
Ж2Г
.__inference_mc_dropout_8_layer_call_fn_2432476
.__inference_mc_dropout_8_layer_call_fn_2432629
.__inference_mc_dropout_8_layer_call_fn_2432642
.__inference_mc_dropout_8_layer_call_fn_2432563└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
"__inference__wrapped_model_2432396┤
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *$в!
К
input         
Є2я
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432675
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432708
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432579
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432595└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┘2╓
/__inference_dropout_tf_68_layer_call_fn_2432713в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_dropout_tf_68_layer_call_and_return_conditional_losses_2432725в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_output_-1_layer_call_fn_2432734в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_output_-1_layer_call_and_return_conditional_losses_2432745в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┘2╓
/__inference_dropout_tf_69_layer_call_fn_2432750в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_dropout_tf_69_layer_call_and_return_conditional_losses_2432762в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_output_layer_call_fn_2432771в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_output_layer_call_and_return_conditional_losses_2432781в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╩B╟
%__inference_signature_wrapper_2432616input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 Н
"__inference__wrapped_model_2432396g.в+
$в!
К
input         
к "/к,
*
output К
output         ж
J__inference_dropout_tf_68_layer_call_and_return_conditional_losses_2432725X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ ~
/__inference_dropout_tf_68_layer_call_fn_2432713K/в,
%в"
 К
inputs         
к "К         ж
J__inference_dropout_tf_69_layer_call_and_return_conditional_losses_2432762X/в,
%в"
 К
inputs         2
к "%в"
К
0         2
Ъ ~
/__inference_dropout_tf_69_layer_call_fn_2432750K/в,
%в"
 К
inputs         2
к "К         2▓
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432579e6в3
,в)
К
input         
p 

 
к "%в"
К
0         
Ъ ▓
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432595e6в3
,в)
К
input         
p

 
к "%в"
К
0         
Ъ │
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432675f7в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ │
I__inference_mc_dropout_8_layer_call_and_return_conditional_losses_2432708f7в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ К
.__inference_mc_dropout_8_layer_call_fn_2432476X6в3
,в)
К
input         
p 

 
к "К         К
.__inference_mc_dropout_8_layer_call_fn_2432563X6в3
,в)
К
input         
p

 
к "К         Л
.__inference_mc_dropout_8_layer_call_fn_2432629Y7в4
-в*
 К
inputs         
p 

 
к "К         Л
.__inference_mc_dropout_8_layer_call_fn_2432642Y7в4
-в*
 К
inputs         
p

 
к "К         ж
F__inference_output_-1_layer_call_and_return_conditional_losses_2432745\/в,
%в"
 К
inputs         
к "%в"
К
0         2
Ъ ~
+__inference_output_-1_layer_call_fn_2432734O/в,
%в"
 К
inputs         
к "К         2г
C__inference_output_layer_call_and_return_conditional_losses_2432781\/в,
%в"
 К
inputs         2
к "%в"
К
0         
Ъ {
(__inference_output_layer_call_fn_2432771O/в,
%в"
 К
inputs         2
к "К         Щ
%__inference_signature_wrapper_2432616p7в4
в 
-к*
(
inputК
input         "/к,
*
output К
output         