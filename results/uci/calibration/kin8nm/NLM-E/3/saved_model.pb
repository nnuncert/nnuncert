��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
delete_old_dirsbool(�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring �
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.1-0-g85c8b2a817f8��
�
3_output_-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*#
shared_name3_output_-1/kernel
y
&3_output_-1/kernel/Read/ReadVariableOpReadVariableOp3_output_-1/kernel*
_output_shapes

:2*
dtype0
x
3_output_-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*!
shared_name3_output_-1/bias
q
$3_output_-1/bias/Read/ReadVariableOpReadVariableOp3_output_-1/bias*
_output_shapes
:2*
dtype0
z
3_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2* 
shared_name3_output/kernel
s
#3_output/kernel/Read/ReadVariableOpReadVariableOp3_output/kernel*
_output_shapes

:2*
dtype0
r
3_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name3_output/bias
k
!3_output/bias/Read/ReadVariableOpReadVariableOp3_output/bias*
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
�
Adam/3_output_-1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2**
shared_nameAdam/3_output_-1/kernel/m
�
-Adam/3_output_-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/3_output_-1/kernel/m*
_output_shapes

:2*
dtype0
�
Adam/3_output_-1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*(
shared_nameAdam/3_output_-1/bias/m

+Adam/3_output_-1/bias/m/Read/ReadVariableOpReadVariableOpAdam/3_output_-1/bias/m*
_output_shapes
:2*
dtype0
�
Adam/3_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/3_output/kernel/m
�
*Adam/3_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/3_output/kernel/m*
_output_shapes

:2*
dtype0
�
Adam/3_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/3_output/bias/m
y
(Adam/3_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/3_output/bias/m*
_output_shapes
:*
dtype0
�
Adam/3_output_-1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2**
shared_nameAdam/3_output_-1/kernel/v
�
-Adam/3_output_-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/3_output_-1/kernel/v*
_output_shapes

:2*
dtype0
�
Adam/3_output_-1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*(
shared_nameAdam/3_output_-1/bias/v

+Adam/3_output_-1/bias/v/Read/ReadVariableOpReadVariableOpAdam/3_output_-1/bias/v*
_output_shapes
:2*
dtype0
�
Adam/3_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdam/3_output/kernel/v
�
*Adam/3_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/3_output/kernel/v*
_output_shapes

:2*
dtype0
�
Adam/3_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/3_output/bias/v
y
(Adam/3_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/3_output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�#
value�#B�# B�#
�
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
�
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
�
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
�
.layer_regularization_losses
/non_trainable_variables
0metrics
trainable_variables
1layer_metrics
regularization_losses

2layers
	variables
^\
VARIABLE_VALUE3_output_-1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE3_output_-1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
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
�
8layer_regularization_losses
9non_trainable_variables
:metrics
trainable_variables
;layer_metrics
regularization_losses

<layers
	variables
[Y
VARIABLE_VALUE3_output/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUE3_output/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
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
�
VARIABLE_VALUEAdam/3_output_-1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/3_output_-1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/3_output/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/3_output/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/3_output_-1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/3_output_-1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/3_output/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/3_output/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_3_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_3_input3_output_-1/kernel3_output_-1/bias3_output/kernel3_output/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_1540399
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&3_output_-1/kernel/Read/ReadVariableOp$3_output_-1/bias/Read/ReadVariableOp#3_output/kernel/Read/ReadVariableOp!3_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp-Adam/3_output_-1/kernel/m/Read/ReadVariableOp+Adam/3_output_-1/bias/m/Read/ReadVariableOp*Adam/3_output/kernel/m/Read/ReadVariableOp(Adam/3_output/bias/m/Read/ReadVariableOp-Adam/3_output_-1/kernel/v/Read/ReadVariableOp+Adam/3_output_-1/bias/v/Read/ReadVariableOp*Adam/3_output/kernel/v/Read/ReadVariableOp(Adam/3_output/bias/v/Read/ReadVariableOpConst*$
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
GPU2*0J 8� *)
f$R"
 __inference__traced_save_1540662
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename3_output_-1/kernel3_output_-1/bias3_output/kernel3_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/3_output_-1/kernel/mAdam/3_output_-1/bias/mAdam/3_output/kernel/mAdam/3_output/bias/mAdam/3_output_-1/kernel/vAdam/3_output_-1/bias/vAdam/3_output/kernel/vAdam/3_output/bias/v*#
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
GPU2*0J 8� *,
f'R%
#__inference__traced_restore_1540741��
�
f
H__inference_dropout_591_layer_call_and_return_conditional_losses_1540541

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�a
�
#__inference__traced_restore_1540741
file_prefix'
#assignvariableop_3_output__1_kernel'
#assignvariableop_1_3_output__1_bias&
"assignvariableop_2_3_output_kernel$
 assignvariableop_3_3_output_bias 
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
-assignvariableop_15_adam_3_output__1_kernel_m/
+assignvariableop_16_adam_3_output__1_bias_m.
*assignvariableop_17_adam_3_output_kernel_m,
(assignvariableop_18_adam_3_output_bias_m1
-assignvariableop_19_adam_3_output__1_kernel_v/
+assignvariableop_20_adam_3_output__1_bias_v.
*assignvariableop_21_adam_3_output_kernel_v,
(assignvariableop_22_adam_3_output_bias_v
identity_24��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
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

Identity�
AssignVariableOpAssignVariableOp#assignvariableop_3_output__1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp#assignvariableop_1_3_output__1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_3_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_3_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp-assignvariableop_15_adam_3_output__1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_3_output__1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_3_output_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_3_output_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp-assignvariableop_19_adam_3_output__1_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_3_output__1_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_3_output_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_3_output_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_229
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23�
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
�	
�
H__inference_3_output_-1_layer_call_and_return_conditional_losses_1540515

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������22
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_3_output_-1_layer_call_fn_1540524

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_3_output_-1_layer_call_and_return_conditional_losses_15402282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_3_output_layer_call_and_return_conditional_losses_1540284

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
g
H__inference_dropout_590_layer_call_and_return_conditional_losses_1540199

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
I
-__inference_dropout_591_layer_call_fn_1540551

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_591_layer_call_and_return_conditional_losses_15402612
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
(__inference_nlm_27_layer_call_fn_1540347	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_nlm_27_layer_call_and_return_conditional_losses_15403362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	3_input
�	
�
E__inference_3_output_layer_call_and_return_conditional_losses_1540561

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540451

inputs,
(output__1_matmul_readvariableop_resource-
)output__1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��3_output/BiasAdd/ReadVariableOp�3_output/MatMul/ReadVariableOp�"3_output_-1/BiasAdd/ReadVariableOp�!3_output_-1/MatMul/ReadVariableOpr
dropout_590/IdentityIdentityinputs*
T0*'
_output_shapes
:���������2
dropout_590/Identity�
!3_output_-1/MatMul/ReadVariableOpReadVariableOp(output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02#
!3_output_-1/MatMul/ReadVariableOp�
3_output_-1/MatMulMatMuldropout_590/Identity:output:0)3_output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
3_output_-1/MatMul�
"3_output_-1/BiasAdd/ReadVariableOpReadVariableOp)output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02$
"3_output_-1/BiasAdd/ReadVariableOp�
3_output_-1/BiasAddBiasAdd3_output_-1/MatMul:product:0*3_output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
3_output_-1/BiasAdd|
3_output_-1/ReluRelu3_output_-1/BiasAdd:output:0*
T0*'
_output_shapes
:���������22
3_output_-1/Relu�
dropout_591/IdentityIdentity3_output_-1/Relu:activations:0*
T0*'
_output_shapes
:���������22
dropout_591/Identity�
3_output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02 
3_output/MatMul/ReadVariableOp�
3_output/MatMulMatMuldropout_591/Identity:output:0&3_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
3_output/MatMul�
3_output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
3_output/BiasAdd/ReadVariableOp�
3_output/BiasAddBiasAdd3_output/MatMul:product:0'3_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
3_output/BiasAdd�
IdentityIdentity3_output/BiasAdd:output:0 ^3_output/BiasAdd/ReadVariableOp^3_output/MatMul/ReadVariableOp#^3_output_-1/BiasAdd/ReadVariableOp"^3_output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2B
3_output/BiasAdd/ReadVariableOp3_output/BiasAdd/ReadVariableOp2@
3_output/MatMul/ReadVariableOp3_output/MatMul/ReadVariableOp2H
"3_output_-1/BiasAdd/ReadVariableOp"3_output_-1/BiasAdd/ReadVariableOp2F
!3_output_-1/MatMul/ReadVariableOp!3_output_-1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540336

inputs
output__1_1540324
output__1_1540326
output_1540330
output_1540332
identity�� 3_output/StatefulPartitionedCall�#3_output_-1/StatefulPartitionedCall�#dropout_590/StatefulPartitionedCall�#dropout_591/StatefulPartitionedCall�
#dropout_590/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_590_layer_call_and_return_conditional_losses_15401992%
#dropout_590/StatefulPartitionedCall�
#3_output_-1/StatefulPartitionedCallStatefulPartitionedCall,dropout_590/StatefulPartitionedCall:output:0output__1_1540324output__1_1540326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_3_output_-1_layer_call_and_return_conditional_losses_15402282%
#3_output_-1/StatefulPartitionedCall�
#dropout_591/StatefulPartitionedCallStatefulPartitionedCall,3_output_-1/StatefulPartitionedCall:output:0$^dropout_590/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_591_layer_call_and_return_conditional_losses_15402562%
#dropout_591/StatefulPartitionedCall�
 3_output/StatefulPartitionedCallStatefulPartitionedCall,dropout_591/StatefulPartitionedCall:output:0output_1540330output_1540332*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_3_output_layer_call_and_return_conditional_losses_15402842"
 3_output/StatefulPartitionedCall�
IdentityIdentity)3_output/StatefulPartitionedCall:output:0!^3_output/StatefulPartitionedCall$^3_output_-1/StatefulPartitionedCall$^dropout_590/StatefulPartitionedCall$^dropout_591/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2D
 3_output/StatefulPartitionedCall 3_output/StatefulPartitionedCall2J
#3_output_-1/StatefulPartitionedCall#3_output_-1/StatefulPartitionedCall2J
#dropout_590/StatefulPartitionedCall#dropout_590/StatefulPartitionedCall2J
#dropout_591/StatefulPartitionedCall#dropout_591/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�5
�	
 __inference__traced_save_1540662
file_prefix1
-savev2_3_output__1_kernel_read_readvariableop/
+savev2_3_output__1_bias_read_readvariableop.
*savev2_3_output_kernel_read_readvariableop,
(savev2_3_output_bias_read_readvariableop(
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
4savev2_adam_3_output__1_kernel_m_read_readvariableop6
2savev2_adam_3_output__1_bias_m_read_readvariableop5
1savev2_adam_3_output_kernel_m_read_readvariableop3
/savev2_adam_3_output_bias_m_read_readvariableop8
4savev2_adam_3_output__1_kernel_v_read_readvariableop6
2savev2_adam_3_output__1_bias_v_read_readvariableop5
1savev2_adam_3_output_kernel_v_read_readvariableop3
/savev2_adam_3_output_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_3_output__1_kernel_read_readvariableop+savev2_3_output__1_bias_read_readvariableop*savev2_3_output_kernel_read_readvariableop(savev2_3_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop4savev2_adam_3_output__1_kernel_m_read_readvariableop2savev2_adam_3_output__1_bias_m_read_readvariableop1savev2_adam_3_output_kernel_m_read_readvariableop/savev2_adam_3_output_bias_m_read_readvariableop4savev2_adam_3_output__1_kernel_v_read_readvariableop2savev2_adam_3_output__1_bias_v_read_readvariableop1savev2_adam_3_output_kernel_v_read_readvariableop/savev2_adam_3_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
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
�

*__inference_3_output_layer_call_fn_1540570

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_3_output_layer_call_and_return_conditional_losses_15402842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1540399	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_15401832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	3_input
�
f
H__inference_dropout_590_layer_call_and_return_conditional_losses_1540204

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
g
H__inference_dropout_591_layer_call_and_return_conditional_losses_1540536

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
(__inference_nlm_27_layer_call_fn_1540376	
input
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_nlm_27_layer_call_and_return_conditional_losses_15403652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	3_input
�
f
H__inference_dropout_590_layer_call_and_return_conditional_losses_1540494

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540317	
input
output__1_1540305
output__1_1540307
output_1540311
output_1540313
identity�� 3_output/StatefulPartitionedCall�#3_output_-1/StatefulPartitionedCall�
dropout_590/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_590_layer_call_and_return_conditional_losses_15402042
dropout_590/PartitionedCall�
#3_output_-1/StatefulPartitionedCallStatefulPartitionedCall$dropout_590/PartitionedCall:output:0output__1_1540305output__1_1540307*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_3_output_-1_layer_call_and_return_conditional_losses_15402282%
#3_output_-1/StatefulPartitionedCall�
dropout_591/PartitionedCallPartitionedCall,3_output_-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_591_layer_call_and_return_conditional_losses_15402612
dropout_591/PartitionedCall�
 3_output/StatefulPartitionedCallStatefulPartitionedCall$dropout_591/PartitionedCall:output:0output_1540311output_1540313*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_3_output_layer_call_and_return_conditional_losses_15402842"
 3_output/StatefulPartitionedCall�
IdentityIdentity)3_output/StatefulPartitionedCall:output:0!^3_output/StatefulPartitionedCall$^3_output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2D
 3_output/StatefulPartitionedCall 3_output/StatefulPartitionedCall2J
#3_output_-1/StatefulPartitionedCall#3_output_-1/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	3_input
�
�
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540301	
input
output__1_1540239
output__1_1540241
output_1540295
output_1540297
identity�� 3_output/StatefulPartitionedCall�#3_output_-1/StatefulPartitionedCall�#dropout_590/StatefulPartitionedCall�#dropout_591/StatefulPartitionedCall�
#dropout_590/StatefulPartitionedCallStatefulPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_590_layer_call_and_return_conditional_losses_15401992%
#dropout_590/StatefulPartitionedCall�
#3_output_-1/StatefulPartitionedCallStatefulPartitionedCall,dropout_590/StatefulPartitionedCall:output:0output__1_1540239output__1_1540241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_3_output_-1_layer_call_and_return_conditional_losses_15402282%
#3_output_-1/StatefulPartitionedCall�
#dropout_591/StatefulPartitionedCallStatefulPartitionedCall,3_output_-1/StatefulPartitionedCall:output:0$^dropout_590/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_591_layer_call_and_return_conditional_losses_15402562%
#dropout_591/StatefulPartitionedCall�
 3_output/StatefulPartitionedCallStatefulPartitionedCall,dropout_591/StatefulPartitionedCall:output:0output_1540295output_1540297*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_3_output_layer_call_and_return_conditional_losses_15402842"
 3_output/StatefulPartitionedCall�
IdentityIdentity)3_output/StatefulPartitionedCall:output:0!^3_output/StatefulPartitionedCall$^3_output_-1/StatefulPartitionedCall$^dropout_590/StatefulPartitionedCall$^dropout_591/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2D
 3_output/StatefulPartitionedCall 3_output/StatefulPartitionedCall2J
#3_output_-1/StatefulPartitionedCall#3_output_-1/StatefulPartitionedCall2J
#dropout_590/StatefulPartitionedCall#dropout_590/StatefulPartitionedCall2J
#dropout_591/StatefulPartitionedCall#dropout_591/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	3_input
�
�
(__inference_nlm_27_layer_call_fn_1540477

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_nlm_27_layer_call_and_return_conditional_losses_15403652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
"__inference__wrapped_model_1540183	
input5
1nlm_27_3_output__1_matmul_readvariableop_resource6
2nlm_27_3_output__1_biasadd_readvariableop_resource2
.nlm_27_3_output_matmul_readvariableop_resource3
/nlm_27_3_output_biasadd_readvariableop_resource
identity��&nlm_27/3_output/BiasAdd/ReadVariableOp�%nlm_27/3_output/MatMul/ReadVariableOp�)nlm_27/3_output_-1/BiasAdd/ReadVariableOp�(nlm_27/3_output_-1/MatMul/ReadVariableOp
nlm_27/dropout_590/IdentityIdentityinput*
T0*'
_output_shapes
:���������2
nlm_27/dropout_590/Identity�
(nlm_27/3_output_-1/MatMul/ReadVariableOpReadVariableOp1nlm_27_3_output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02*
(nlm_27/3_output_-1/MatMul/ReadVariableOp�
nlm_27/3_output_-1/MatMulMatMul$nlm_27/dropout_590/Identity:output:00nlm_27/3_output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
nlm_27/3_output_-1/MatMul�
)nlm_27/3_output_-1/BiasAdd/ReadVariableOpReadVariableOp2nlm_27_3_output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)nlm_27/3_output_-1/BiasAdd/ReadVariableOp�
nlm_27/3_output_-1/BiasAddBiasAdd#nlm_27/3_output_-1/MatMul:product:01nlm_27/3_output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
nlm_27/3_output_-1/BiasAdd�
nlm_27/3_output_-1/ReluRelu#nlm_27/3_output_-1/BiasAdd:output:0*
T0*'
_output_shapes
:���������22
nlm_27/3_output_-1/Relu�
nlm_27/dropout_591/IdentityIdentity%nlm_27/3_output_-1/Relu:activations:0*
T0*'
_output_shapes
:���������22
nlm_27/dropout_591/Identity�
%nlm_27/3_output/MatMul/ReadVariableOpReadVariableOp.nlm_27_3_output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02'
%nlm_27/3_output/MatMul/ReadVariableOp�
nlm_27/3_output/MatMulMatMul$nlm_27/dropout_591/Identity:output:0-nlm_27/3_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
nlm_27/3_output/MatMul�
&nlm_27/3_output/BiasAdd/ReadVariableOpReadVariableOp/nlm_27_3_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&nlm_27/3_output/BiasAdd/ReadVariableOp�
nlm_27/3_output/BiasAddBiasAdd nlm_27/3_output/MatMul:product:0.nlm_27/3_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
nlm_27/3_output/BiasAdd�
IdentityIdentity nlm_27/3_output/BiasAdd:output:0'^nlm_27/3_output/BiasAdd/ReadVariableOp&^nlm_27/3_output/MatMul/ReadVariableOp*^nlm_27/3_output_-1/BiasAdd/ReadVariableOp)^nlm_27/3_output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2P
&nlm_27/3_output/BiasAdd/ReadVariableOp&nlm_27/3_output/BiasAdd/ReadVariableOp2N
%nlm_27/3_output/MatMul/ReadVariableOp%nlm_27/3_output/MatMul/ReadVariableOp2V
)nlm_27/3_output_-1/BiasAdd/ReadVariableOp)nlm_27/3_output_-1/BiasAdd/ReadVariableOp2T
(nlm_27/3_output_-1/MatMul/ReadVariableOp(nlm_27/3_output_-1/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	3_input
�	
�
H__inference_3_output_-1_layer_call_and_return_conditional_losses_1540228

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������22
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540432

inputs,
(output__1_matmul_readvariableop_resource-
)output__1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��3_output/BiasAdd/ReadVariableOp�3_output/MatMul/ReadVariableOp�"3_output_-1/BiasAdd/ReadVariableOp�!3_output_-1/MatMul/ReadVariableOp{
dropout_590/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_590/dropout/Const�
dropout_590/dropout/MulMulinputs"dropout_590/dropout/Const:output:0*
T0*'
_output_shapes
:���������2
dropout_590/dropout/Mull
dropout_590/dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout_590/dropout/Shape�
0dropout_590/dropout/random_uniform/RandomUniformRandomUniform"dropout_590/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype022
0dropout_590/dropout/random_uniform/RandomUniform�
"dropout_590/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout_590/dropout/GreaterEqual/y�
 dropout_590/dropout/GreaterEqualGreaterEqual9dropout_590/dropout/random_uniform/RandomUniform:output:0+dropout_590/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2"
 dropout_590/dropout/GreaterEqual�
dropout_590/dropout/CastCast$dropout_590/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2
dropout_590/dropout/Cast�
dropout_590/dropout/Mul_1Muldropout_590/dropout/Mul:z:0dropout_590/dropout/Cast:y:0*
T0*'
_output_shapes
:���������2
dropout_590/dropout/Mul_1�
!3_output_-1/MatMul/ReadVariableOpReadVariableOp(output__1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02#
!3_output_-1/MatMul/ReadVariableOp�
3_output_-1/MatMulMatMuldropout_590/dropout/Mul_1:z:0)3_output_-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
3_output_-1/MatMul�
"3_output_-1/BiasAdd/ReadVariableOpReadVariableOp)output__1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02$
"3_output_-1/BiasAdd/ReadVariableOp�
3_output_-1/BiasAddBiasAdd3_output_-1/MatMul:product:0*3_output_-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
3_output_-1/BiasAdd|
3_output_-1/ReluRelu3_output_-1/BiasAdd:output:0*
T0*'
_output_shapes
:���������22
3_output_-1/Relu{
dropout_591/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_591/dropout/Const�
dropout_591/dropout/MulMul3_output_-1/Relu:activations:0"dropout_591/dropout/Const:output:0*
T0*'
_output_shapes
:���������22
dropout_591/dropout/Mul�
dropout_591/dropout/ShapeShape3_output_-1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_591/dropout/Shape�
0dropout_591/dropout/random_uniform/RandomUniformRandomUniform"dropout_591/dropout/Shape:output:0*
T0*'
_output_shapes
:���������2*
dtype022
0dropout_591/dropout/random_uniform/RandomUniform�
"dropout_591/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dropout_591/dropout/GreaterEqual/y�
 dropout_591/dropout/GreaterEqualGreaterEqual9dropout_591/dropout/random_uniform/RandomUniform:output:0+dropout_591/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������22"
 dropout_591/dropout/GreaterEqual�
dropout_591/dropout/CastCast$dropout_591/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������22
dropout_591/dropout/Cast�
dropout_591/dropout/Mul_1Muldropout_591/dropout/Mul:z:0dropout_591/dropout/Cast:y:0*
T0*'
_output_shapes
:���������22
dropout_591/dropout/Mul_1�
3_output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02 
3_output/MatMul/ReadVariableOp�
3_output/MatMulMatMuldropout_591/dropout/Mul_1:z:0&3_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
3_output/MatMul�
3_output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
3_output/BiasAdd/ReadVariableOp�
3_output/BiasAddBiasAdd3_output/MatMul:product:0'3_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
3_output/BiasAdd�
IdentityIdentity3_output/BiasAdd:output:0 ^3_output/BiasAdd/ReadVariableOp^3_output/MatMul/ReadVariableOp#^3_output_-1/BiasAdd/ReadVariableOp"^3_output_-1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2B
3_output/BiasAdd/ReadVariableOp3_output/BiasAdd/ReadVariableOp2@
3_output/MatMul/ReadVariableOp3_output/MatMul/ReadVariableOp2H
"3_output_-1/BiasAdd/ReadVariableOp"3_output_-1/BiasAdd/ReadVariableOp2F
!3_output_-1/MatMul/ReadVariableOp!3_output_-1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
g
H__inference_dropout_591_layer_call_and_return_conditional_losses_1540256

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
f
-__inference_dropout_590_layer_call_fn_1540499

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_590_layer_call_and_return_conditional_losses_15401992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
-__inference_dropout_591_layer_call_fn_1540546

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_591_layer_call_and_return_conditional_losses_15402562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*&
_input_shapes
:���������222
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
(__inference_nlm_27_layer_call_fn_1540464

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_nlm_27_layer_call_and_return_conditional_losses_15403362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540365

inputs
output__1_1540353
output__1_1540355
output_1540359
output_1540361
identity�� 3_output/StatefulPartitionedCall�#3_output_-1/StatefulPartitionedCall�
dropout_590/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_590_layer_call_and_return_conditional_losses_15402042
dropout_590/PartitionedCall�
#3_output_-1/StatefulPartitionedCallStatefulPartitionedCall$dropout_590/PartitionedCall:output:0output__1_1540353output__1_1540355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_3_output_-1_layer_call_and_return_conditional_losses_15402282%
#3_output_-1/StatefulPartitionedCall�
dropout_591/PartitionedCallPartitionedCall,3_output_-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_591_layer_call_and_return_conditional_losses_15402612
dropout_591/PartitionedCall�
 3_output/StatefulPartitionedCallStatefulPartitionedCall$dropout_591/PartitionedCall:output:0output_1540359output_1540361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_3_output_layer_call_and_return_conditional_losses_15402842"
 3_output/StatefulPartitionedCall�
IdentityIdentity)3_output/StatefulPartitionedCall:output:0!^3_output/StatefulPartitionedCall$^3_output_-1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2D
 3_output/StatefulPartitionedCall 3_output/StatefulPartitionedCall2J
#3_output_-1/StatefulPartitionedCall#3_output_-1/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
H__inference_dropout_591_layer_call_and_return_conditional_losses_1540261

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
g
H__inference_dropout_590_layer_call_and_return_conditional_losses_1540489

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
I
-__inference_dropout_590_layer_call_fn_1540504

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_dropout_590_layer_call_and_return_conditional_losses_15402042
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
3_input0
serving_default_3_input:0���������<
3_output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�'
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
^_default_save_signature"�$
_tf_keras_network�${"class_name": "NLM", "name": "nlm_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "nlm_27", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "3_input"}, "name": "3_input", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_590", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}, "name": "dropout_590", "inbound_nodes": [[["3_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "3_output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "3_output_-1", "inbound_nodes": [[["dropout_590", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_591", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}, "name": "dropout_591", "inbound_nodes": [[["3_output_-1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "3_output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "3_output", "inbound_nodes": [[["dropout_591", 0, 0, {}]]]}], "input_layers": [["3_input", 0, 0]], "output_layers": [["3_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "NLM", "config": {"name": "nlm_27", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "3_input"}, "name": "3_input", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_590", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}, "name": "dropout_590", "inbound_nodes": [[["3_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "3_output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "3_output_-1", "inbound_nodes": [[["dropout_590", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_591", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}, "name": "dropout_591", "inbound_nodes": [[["3_output_-1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "3_output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "3_output", "inbound_nodes": [[["dropout_591", 0, 0, {}]]]}], "input_layers": [["3_input", 0, 0]], "output_layers": [["3_output", 0, 0]]}}, "training_config": {"loss": "nll", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "3_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "3_input"}}
�
trainable_variables
regularization_losses
	variables
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_590", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_590", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "3_output_-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "3_output_-1", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
�
trainable_variables
regularization_losses
	variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_591", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_591", "trainable": true, "dtype": "float32", "rate": 0, "noise_shape": null, "seed": null}}
�

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
*e&call_and_return_all_conditional_losses
f__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "3_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "3_output", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
*
"tau2"
trackable_dict_wrapper
(
#1"
trackable_tuple_wrapper
�
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
�
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
�
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
$:"223_output_-1/kernel
:223_output_-1/bias
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
�
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
�
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
!:223_output/kernel
:23_output/bias
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
�
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
�
	Ftotal
	Gcount
H	variables
I	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	Jtotal
	Kcount
L
_fn_kwargs
M	variables
N	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
�
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
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
):'22Adam/3_output_-1/kernel/m
#:!22Adam/3_output_-1/bias/m
&:$22Adam/3_output/kernel/m
 :2Adam/3_output/bias/m
):'22Adam/3_output_-1/kernel/v
#:!22Adam/3_output_-1/bias/v
&:$22Adam/3_output/kernel/v
 :2Adam/3_output/bias/v
�2�
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540301
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540451
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540432
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540317�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_nlm_27_layer_call_fn_1540347
(__inference_nlm_27_layer_call_fn_1540464
(__inference_nlm_27_layer_call_fn_1540477
(__inference_nlm_27_layer_call_fn_1540376�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_1540183�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
3_input���������
�2�
H__inference_dropout_590_layer_call_and_return_conditional_losses_1540489
H__inference_dropout_590_layer_call_and_return_conditional_losses_1540494�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_dropout_590_layer_call_fn_1540504
-__inference_dropout_590_layer_call_fn_1540499�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_3_output_-1_layer_call_and_return_conditional_losses_1540515�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_3_output_-1_layer_call_fn_1540524�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_dropout_591_layer_call_and_return_conditional_losses_1540541
H__inference_dropout_591_layer_call_and_return_conditional_losses_1540536�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_dropout_591_layer_call_fn_1540551
-__inference_dropout_591_layer_call_fn_1540546�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_3_output_layer_call_and_return_conditional_losses_1540561�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_3_output_layer_call_fn_1540570�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_15403993_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
H__inference_3_output_-1_layer_call_and_return_conditional_losses_1540515\/�,
%�"
 �
inputs���������
� "%�"
�
0���������2
� �
-__inference_3_output_-1_layer_call_fn_1540524O/�,
%�"
 �
inputs���������
� "����������2�
E__inference_3_output_layer_call_and_return_conditional_losses_1540561\/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� }
*__inference_3_output_layer_call_fn_1540570O/�,
%�"
 �
inputs���������2
� "�����������
"__inference__wrapped_model_1540183m0�-
&�#
!�
3_input���������
� "3�0
.
3_output"�
3_output����������
H__inference_dropout_590_layer_call_and_return_conditional_losses_1540489\3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
H__inference_dropout_590_layer_call_and_return_conditional_losses_1540494\3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
-__inference_dropout_590_layer_call_fn_1540499O3�0
)�&
 �
inputs���������
p
� "�����������
-__inference_dropout_590_layer_call_fn_1540504O3�0
)�&
 �
inputs���������
p 
� "�����������
H__inference_dropout_591_layer_call_and_return_conditional_losses_1540536\3�0
)�&
 �
inputs���������2
p
� "%�"
�
0���������2
� �
H__inference_dropout_591_layer_call_and_return_conditional_losses_1540541\3�0
)�&
 �
inputs���������2
p 
� "%�"
�
0���������2
� �
-__inference_dropout_591_layer_call_fn_1540546O3�0
)�&
 �
inputs���������2
p
� "����������2�
-__inference_dropout_591_layer_call_fn_1540551O3�0
)�&
 �
inputs���������2
p 
� "����������2�
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540301g8�5
.�+
!�
3_input���������
p

 
� "%�"
�
0���������
� �
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540317g8�5
.�+
!�
3_input���������
p 

 
� "%�"
�
0���������
� �
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540432f7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
C__inference_nlm_27_layer_call_and_return_conditional_losses_1540451f7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
(__inference_nlm_27_layer_call_fn_1540347Z8�5
.�+
!�
3_input���������
p

 
� "�����������
(__inference_nlm_27_layer_call_fn_1540376Z8�5
.�+
!�
3_input���������
p 

 
� "�����������
(__inference_nlm_27_layer_call_fn_1540464Y7�4
-�*
 �
inputs���������
p

 
� "�����������
(__inference_nlm_27_layer_call_fn_1540477Y7�4
-�*
 �
inputs���������
p 

 
� "�����������
%__inference_signature_wrapper_1540399x;�8
� 
1�.
,
3_input!�
3_input���������"3�0
.
3_output"�
3_output���������