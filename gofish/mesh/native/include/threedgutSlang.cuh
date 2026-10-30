// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#include <slang-cuda-prelude.h>

struct shRadiativeParticle_Parameters_0
{
    FixedArray<float3 , 16>  sphCoefficients_0;
};

__device__ shRadiativeParticle_Parameters_0 shRadiativeParticle_Parameters_x24_syn_dzero_0()
{
    shRadiativeParticle_Parameters_0 result_0;
    float3  _S1 = make_float3 (0.0f);
    (&result_0)->sphCoefficients_0[int(0)] = _S1;
    (&result_0)->sphCoefficients_0[int(1)] = _S1;
    (&result_0)->sphCoefficients_0[int(2)] = _S1;
    (&result_0)->sphCoefficients_0[int(3)] = _S1;
    (&result_0)->sphCoefficients_0[int(4)] = _S1;
    (&result_0)->sphCoefficients_0[int(5)] = _S1;
    (&result_0)->sphCoefficients_0[int(6)] = _S1;
    (&result_0)->sphCoefficients_0[int(7)] = _S1;
    (&result_0)->sphCoefficients_0[int(8)] = _S1;
    (&result_0)->sphCoefficients_0[int(9)] = _S1;
    (&result_0)->sphCoefficients_0[int(10)] = _S1;
    (&result_0)->sphCoefficients_0[int(11)] = _S1;
    (&result_0)->sphCoefficients_0[int(12)] = _S1;
    (&result_0)->sphCoefficients_0[int(13)] = _S1;
    (&result_0)->sphCoefficients_0[int(14)] = _S1;
    (&result_0)->sphCoefficients_0[int(15)] = _S1;
    return result_0;
}

__device__ shRadiativeParticle_Parameters_0 shRadiativeParticle_Parameters_x24_syn_dadd_0(shRadiativeParticle_Parameters_0 * SLANG_anonymous_0_0, shRadiativeParticle_Parameters_0 * SLANG_anonymous_1_0)
{
    shRadiativeParticle_Parameters_0 result_1;
    (&result_1)->sphCoefficients_0[int(0)] = SLANG_anonymous_0_0->sphCoefficients_0[int(0)] + SLANG_anonymous_1_0->sphCoefficients_0[int(0)];
    (&result_1)->sphCoefficients_0[int(1)] = SLANG_anonymous_0_0->sphCoefficients_0[int(1)] + SLANG_anonymous_1_0->sphCoefficients_0[int(1)];
    (&result_1)->sphCoefficients_0[int(2)] = SLANG_anonymous_0_0->sphCoefficients_0[int(2)] + SLANG_anonymous_1_0->sphCoefficients_0[int(2)];
    (&result_1)->sphCoefficients_0[int(3)] = SLANG_anonymous_0_0->sphCoefficients_0[int(3)] + SLANG_anonymous_1_0->sphCoefficients_0[int(3)];
    (&result_1)->sphCoefficients_0[int(4)] = SLANG_anonymous_0_0->sphCoefficients_0[int(4)] + SLANG_anonymous_1_0->sphCoefficients_0[int(4)];
    (&result_1)->sphCoefficients_0[int(5)] = SLANG_anonymous_0_0->sphCoefficients_0[int(5)] + SLANG_anonymous_1_0->sphCoefficients_0[int(5)];
    (&result_1)->sphCoefficients_0[int(6)] = SLANG_anonymous_0_0->sphCoefficients_0[int(6)] + SLANG_anonymous_1_0->sphCoefficients_0[int(6)];
    (&result_1)->sphCoefficients_0[int(7)] = SLANG_anonymous_0_0->sphCoefficients_0[int(7)] + SLANG_anonymous_1_0->sphCoefficients_0[int(7)];
    (&result_1)->sphCoefficients_0[int(8)] = SLANG_anonymous_0_0->sphCoefficients_0[int(8)] + SLANG_anonymous_1_0->sphCoefficients_0[int(8)];
    (&result_1)->sphCoefficients_0[int(9)] = SLANG_anonymous_0_0->sphCoefficients_0[int(9)] + SLANG_anonymous_1_0->sphCoefficients_0[int(9)];
    (&result_1)->sphCoefficients_0[int(10)] = SLANG_anonymous_0_0->sphCoefficients_0[int(10)] + SLANG_anonymous_1_0->sphCoefficients_0[int(10)];
    (&result_1)->sphCoefficients_0[int(11)] = SLANG_anonymous_0_0->sphCoefficients_0[int(11)] + SLANG_anonymous_1_0->sphCoefficients_0[int(11)];
    (&result_1)->sphCoefficients_0[int(12)] = SLANG_anonymous_0_0->sphCoefficients_0[int(12)] + SLANG_anonymous_1_0->sphCoefficients_0[int(12)];
    (&result_1)->sphCoefficients_0[int(13)] = SLANG_anonymous_0_0->sphCoefficients_0[int(13)] + SLANG_anonymous_1_0->sphCoefficients_0[int(13)];
    (&result_1)->sphCoefficients_0[int(14)] = SLANG_anonymous_0_0->sphCoefficients_0[int(14)] + SLANG_anonymous_1_0->sphCoefficients_0[int(14)];
    (&result_1)->sphCoefficients_0[int(15)] = SLANG_anonymous_0_0->sphCoefficients_0[int(15)] + SLANG_anonymous_1_0->sphCoefficients_0[int(15)];
    return result_1;
}

struct gaussianParticle_Parameters_0
{
    float3  position_0;
    Matrix<float, 3, 3>  factor_0;
    Matrix<float, 3, 3>  inverseFactor_0;
    float density_0;
};

__device__ gaussianParticle_Parameters_0 gaussianParticle_Parameters_x24_syn_dzero_0()
{
    gaussianParticle_Parameters_0 result_2;
    (&result_2)->position_0 = make_float3 (0.0f);
    Matrix<float, 3, 3>  _S2 = makeMatrix<float, 3, 3> (0.0f);
    (&result_2)->factor_0 = _S2;
    (&result_2)->inverseFactor_0 = _S2;
    (&result_2)->density_0 = 0.0f;
    return result_2;
}

struct gaussianParticle_RawParameters_0
{
    float3  position_1;
    float density_1;
    float4  quaternion_0;
    float3  scale_0;
    float padding_0;
    Matrix<float, 3, 3>  factor_1;
    Matrix<float, 3, 3>  inverseFactor_1;
    float2  fullShapePadding_0;
};

__device__ gaussianParticle_RawParameters_0 gaussianParticle_RawParameters_x24_syn_dzero_0()
{
    gaussianParticle_RawParameters_0 result_3;
    float3  _S3 = make_float3 (0.0f);
    (&result_3)->position_1 = _S3;
    (&result_3)->density_1 = 0.0f;
    (&result_3)->quaternion_0 = make_float4 (0.0f);
    (&result_3)->scale_0 = _S3;
    (&result_3)->padding_0 = 0.0f;
    Matrix<float, 3, 3>  _S4 = makeMatrix<float, 3, 3> (0.0f);
    (&result_3)->factor_1 = _S4;
    (&result_3)->inverseFactor_1 = _S4;
    (&result_3)->fullShapePadding_0 = make_float2 (0.0f);
    return result_3;
}

__device__ gaussianParticle_Parameters_0 gaussianParticle_Parameters_x24init_0(float3  position_2, Matrix<float, 3, 3>  factor_2, Matrix<float, 3, 3>  inverseFactor_2, float density_2)
{
    gaussianParticle_Parameters_0 _S5;
    (&_S5)->position_0 = position_2;
    (&_S5)->factor_0 = factor_2;
    (&_S5)->inverseFactor_0 = inverseFactor_2;
    (&_S5)->density_0 = density_2;
    return _S5;
}

struct gaussianParticle_RawParametersBuffer_0
{
    gaussianParticle_RawParameters_0 * _dataPtr_0;
    gaussianParticle_RawParameters_0 * _gradPtr_0;
    bool exclusiveGradient_0;
};

struct gaussianParticle_CommonParameters_0
{
    gaussianParticle_RawParametersBuffer_0 parametersBuffer_0;
};

__device__ gaussianParticle_Parameters_0 particleDensityParameters(uint particleIdx_0, gaussianParticle_CommonParameters_0 commonParameters_0)
{
    gaussianParticle_RawParameters_0 * _S6 = commonParameters_0.parametersBuffer_0._dataPtr_0 + particleIdx_0;
    return gaussianParticle_Parameters_x24init_0((*_S6).position_1, (*_S6).factor_1, (*_S6).inverseFactor_1, (*_S6).density_1);
}

struct DiffPair_matrixx3Cfloatx2C3x2C3x3E_0
{
    Matrix<float, 3, 3>  primal_0;
    Matrix<float, 3, 3>  differential_0;
};

struct DiffPair_vectorx3Cfloatx2C3x3E_0
{
    float3  primal_0;
    float3  differential_0;
};

__device__ void _d_mul_0(DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * left_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * right_0, float3  dOut_0)
{
    float _S7 = (*left_0).primal_0.rows[int(0)].x * dOut_0.x;
    Matrix<float, 3, 3>  left_d_result_0;
    *&(((&left_d_result_0)->rows + (int(0)))->x) = (*right_0).primal_0.x * dOut_0.x;
    float sum_0 = _S7 + (*left_0).primal_0.rows[int(1)].x * dOut_0.y;
    *&(((&left_d_result_0)->rows + (int(1)))->x) = (*right_0).primal_0.x * dOut_0.y;
    float sum_1 = sum_0 + (*left_0).primal_0.rows[int(2)].x * dOut_0.z;
    *&(((&left_d_result_0)->rows + (int(2)))->x) = (*right_0).primal_0.x * dOut_0.z;
    float3  right_d_result_0;
    *&((&right_d_result_0)->x) = sum_1;
    float _S8 = (*left_0).primal_0.rows[int(0)].y * dOut_0.x;
    *&(((&left_d_result_0)->rows + (int(0)))->y) = (*right_0).primal_0.y * dOut_0.x;
    float sum_2 = _S8 + (*left_0).primal_0.rows[int(1)].y * dOut_0.y;
    *&(((&left_d_result_0)->rows + (int(1)))->y) = (*right_0).primal_0.y * dOut_0.y;
    float sum_3 = sum_2 + (*left_0).primal_0.rows[int(2)].y * dOut_0.z;
    *&(((&left_d_result_0)->rows + (int(2)))->y) = (*right_0).primal_0.y * dOut_0.z;
    *&((&right_d_result_0)->y) = sum_3;
    float _S9 = (*left_0).primal_0.rows[int(0)].z * dOut_0.x;
    *&(((&left_d_result_0)->rows + (int(0)))->z) = (*right_0).primal_0.z * dOut_0.x;
    float sum_4 = _S9 + (*left_0).primal_0.rows[int(1)].z * dOut_0.y;
    *&(((&left_d_result_0)->rows + (int(1)))->z) = (*right_0).primal_0.z * dOut_0.y;
    float sum_5 = sum_4 + (*left_0).primal_0.rows[int(2)].z * dOut_0.z;
    *&(((&left_d_result_0)->rows + (int(2)))->z) = (*right_0).primal_0.z * dOut_0.z;
    *&((&right_d_result_0)->z) = sum_5;
    left_0->primal_0 = (*left_0).primal_0;
    left_0->differential_0 = left_d_result_0;
    right_0->primal_0 = (*right_0).primal_0;
    right_0->differential_0 = right_d_result_0;
    return;
}

__device__ float3  mul_0(Matrix<float, 3, 3>  left_1, float3  right_1)
{
    float3  result_4;
    int i_0 = int(0);
    for(;;)
    {
        if(i_0 < int(3))
        {
        }
        else
        {
            break;
        }
        int j_0 = int(0);
        float sum_6 = 0.0f;
        for(;;)
        {
            if(j_0 < int(3))
            {
            }
            else
            {
                break;
            }
            float sum_7 = sum_6 + _slang_vector_get_element(left_1.rows[i_0], j_0) * _slang_vector_get_element(right_1, j_0);
            j_0 = j_0 + int(1);
            sum_6 = sum_7;
        }
        *_slang_vector_get_element_ptr(&result_4, i_0) = sum_6;
        i_0 = i_0 + int(1);
    }
    return result_4;
}

struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};

__device__ void _d_max_0(DiffPair_float_0 * dpx_0, DiffPair_float_0 * dpy_0, float dOut_1)
{
    DiffPair_float_0 _S10 = *dpx_0;
    float _S11;
    if(((*dpx_0).primal_0) > ((*dpy_0).primal_0))
    {
        _S11 = dOut_1;
    }
    else
    {
        if(((*dpx_0).primal_0) < ((*dpy_0).primal_0))
        {
            _S11 = 0.0f;
        }
        else
        {
            _S11 = 0.5f * dOut_1;
        }
    }
    dpx_0->primal_0 = _S10.primal_0;
    dpx_0->differential_0 = _S11;
    DiffPair_float_0 _S12 = *dpy_0;
    if(((*dpy_0).primal_0) > (_S10.primal_0))
    {
        _S11 = dOut_1;
    }
    else
    {
        if(((*dpy_0).primal_0) < ((*dpx_0).primal_0))
        {
            _S11 = 0.0f;
        }
        else
        {
            _S11 = 0.5f * dOut_1;
        }
    }
    dpy_0->primal_0 = _S12.primal_0;
    dpy_0->differential_0 = _S11;
    return;
}

__device__ void _d_sqrt_0(DiffPair_float_0 * dpx_1, float dOut_2)
{
    float _S13 = 0.5f / (F32_sqrt(((F32_max((1.00000001168609742e-07f), ((*dpx_1).primal_0)))))) * dOut_2;
    dpx_1->primal_0 = (*dpx_1).primal_0;
    dpx_1->differential_0 = _S13;
    return;
}

__device__ void _d_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_2, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_1, float dOut_3)
{
    float3  x_d_result_0;
    *&((&x_d_result_0)->x) = (*dpy_1).primal_0.x * dOut_3;
    float3  y_d_result_0;
    *&((&y_d_result_0)->x) = (*dpx_2).primal_0.x * dOut_3;
    *&((&x_d_result_0)->y) = (*dpy_1).primal_0.y * dOut_3;
    *&((&y_d_result_0)->y) = (*dpx_2).primal_0.y * dOut_3;
    *&((&x_d_result_0)->z) = (*dpy_1).primal_0.z * dOut_3;
    *&((&y_d_result_0)->z) = (*dpx_2).primal_0.z * dOut_3;
    dpx_2->primal_0 = (*dpx_2).primal_0;
    dpx_2->differential_0 = x_d_result_0;
    dpy_1->primal_0 = (*dpy_1).primal_0;
    dpy_1->differential_0 = y_d_result_0;
    return;
}

__device__ float dot_0(float3  x_0, float3  y_0)
{
    int i_1 = int(0);
    float result_5 = 0.0f;
    for(;;)
    {
        if(i_1 < int(3))
        {
        }
        else
        {
            break;
        }
        float result_6 = result_5 + _slang_vector_get_element(x_0, i_1) * _slang_vector_get_element(y_0, i_1);
        i_1 = i_1 + int(1);
        result_5 = result_6;
    }
    return result_5;
}

__device__ float length_0(float3  x_1)
{
    return (F32_sqrt((dot_0(x_1, x_1))));
}

__device__ float3  normalize_0(float3  x_2)
{
    return x_2 / make_float3 (length_0(x_2));
}

__device__ void _d_cross_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * a_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * b_0, float3  dOut_4)
{
    float _S14 = dOut_4.y;
    float _S15 = dOut_4.z;
    float _S16 = dOut_4.x;
    float _S17 = (*a_0).primal_0.z * _S14 + - (*a_0).primal_0.y * _S15;
    float _S18 = - (*a_0).primal_0.z * _S16 + (*a_0).primal_0.x * _S15;
    float _S19 = (*a_0).primal_0.y * _S16 + - (*a_0).primal_0.x * _S14;
    float3  _S20 = make_float3 (- (*b_0).primal_0.z * _S14 + (*b_0).primal_0.y * _S15, (*b_0).primal_0.z * _S16 + - (*b_0).primal_0.x * _S15, - (*b_0).primal_0.y * _S16 + (*b_0).primal_0.x * _S14);
    a_0->primal_0 = (*a_0).primal_0;
    a_0->differential_0 = _S20;
    float3  _S21 = make_float3 (_S17, _S18, _S19);
    b_0->primal_0 = (*b_0).primal_0;
    b_0->differential_0 = _S21;
    return;
}

__device__ float3  cross_0(float3  left_2, float3  right_2)
{
    float _S22 = left_2.y;
    float _S23 = right_2.z;
    float _S24 = left_2.z;
    float _S25 = right_2.y;
    float _S26 = right_2.x;
    float _S27 = left_2.x;
    return make_float3 (_S22 * _S23 - _S24 * _S25, _S24 * _S26 - _S27 * _S23, _S27 * _S25 - _S22 * _S26);
}

__device__ void _d_exp_0(DiffPair_float_0 * dpx_3, float dOut_5)
{
    float _S28 = (F32_exp(((*dpx_3).primal_0))) * dOut_5;
    dpx_3->primal_0 = (*dpx_3).primal_0;
    dpx_3->differential_0 = _S28;
    return;
}

__device__ void _d_min_0(DiffPair_float_0 * dpx_4, DiffPair_float_0 * dpy_2, float dOut_6)
{
    DiffPair_float_0 _S29 = *dpx_4;
    float _S30;
    if(((*dpx_4).primal_0) < ((*dpy_2).primal_0))
    {
        _S30 = dOut_6;
    }
    else
    {
        if(((*dpx_4).primal_0) > ((*dpy_2).primal_0))
        {
            _S30 = 0.0f;
        }
        else
        {
            _S30 = 0.5f * dOut_6;
        }
    }
    dpx_4->primal_0 = _S29.primal_0;
    dpx_4->differential_0 = _S30;
    DiffPair_float_0 _S31 = *dpy_2;
    if(((*dpy_2).primal_0) < (_S29.primal_0))
    {
        _S30 = dOut_6;
    }
    else
    {
        if(((*dpy_2).primal_0) > ((*dpx_4).primal_0))
        {
            _S30 = 0.0f;
        }
        else
        {
            _S30 = 0.5f * dOut_6;
        }
    }
    dpy_2->primal_0 = _S31.primal_0;
    dpy_2->differential_0 = _S30;
    return;
}

__device__ Matrix<float, 3, 3>  transpose_0(Matrix<float, 3, 3>  x_3)
{
    Matrix<float, 3, 3>  result_7;
    int r_0 = int(0);
    for(;;)
    {
        if(r_0 < int(3))
        {
        }
        else
        {
            break;
        }
        int c_0 = int(0);
        for(;;)
        {
            if(c_0 < int(3))
            {
            }
            else
            {
                break;
            }
            *_slang_vector_get_element_ptr(((&result_7)->rows + (r_0)), c_0) = _slang_vector_get_element(x_3.rows[c_0], r_0);
            c_0 = c_0 + int(1);
        }
        r_0 = r_0 + int(1);
    }
    return result_7;
}

__device__ bool particleDensityHit(float3  rayOrigin_0, float3  rayDirection_0, gaussianParticle_Parameters_0 parameters_0, float * alpha_0, float * depth_0, float3  * canonicalIntersection_0, bool enableNormal_0, float3  * normal_0)
{
    float3  canonicalRayOrigin_0 = mul_0(parameters_0.inverseFactor_0, rayOrigin_0 - parameters_0.position_0);
    float3  grdu_0 = mul_0(parameters_0.inverseFactor_0, rayDirection_0);
    float3  canonicalRayDirection_0 = normalize_0(grdu_0);
    float3  _S32 = cross_0(canonicalRayDirection_0, canonicalRayOrigin_0);
    float _S33 = (F32_exp((-0.5f * dot_0(_S32, _S32))));
    *alpha_0 = (F32_min((0.99000000953674316f), (_S33 * parameters_0.density_0)));
    bool acceptHit_0;
    if(_S33 > 0.01130000036209822f)
    {
        acceptHit_0 = (*alpha_0) > 0.00392156885936856f;
    }
    else
    {
        acceptHit_0 = false;
    }
    if(acceptHit_0)
    {
        float _S34 = - dot_0(canonicalRayOrigin_0, grdu_0) / dot_0(grdu_0, grdu_0);
        *depth_0 = _S34;
        *canonicalIntersection_0 = canonicalRayOrigin_0 + make_float3 (_S34) * grdu_0;
        if(enableNormal_0)
        {
            *normal_0 = - normalize_0(mul_0(transpose_0(parameters_0.inverseFactor_0), canonicalRayDirection_0));
        }
    }
    return acceptHit_0;
}

__device__ void _d_clamp_0(DiffPair_float_0 * dpx_5, DiffPair_float_0 * dpMin_0, DiffPair_float_0 * dpMax_0, float dOut_7)
{
    DiffPair_float_0 _S35 = *dpx_5;
    bool _S36;
    if(((*dpx_5).primal_0) >= ((*dpMin_0).primal_0))
    {
        _S36 = ((*dpx_5).primal_0) <= ((*dpMax_0).primal_0);
    }
    else
    {
        _S36 = false;
    }
    float _S37;
    if(_S36)
    {
        _S37 = dOut_7;
    }
    else
    {
        _S37 = 0.0f;
    }
    dpx_5->primal_0 = _S35.primal_0;
    dpx_5->differential_0 = _S37;
    DiffPair_float_0 _S38 = *dpMin_0;
    if((_S35.primal_0) < ((*dpMin_0).primal_0))
    {
        _S37 = dOut_7;
    }
    else
    {
        _S37 = 0.0f;
    }
    dpMin_0->primal_0 = _S38.primal_0;
    dpMin_0->differential_0 = _S37;
    DiffPair_float_0 _S39 = *dpMax_0;
    if(((*dpx_5).primal_0) > ((*dpMax_0).primal_0))
    {
        _S37 = dOut_7;
    }
    else
    {
        _S37 = 0.0f;
    }
    dpMax_0->primal_0 = _S39.primal_0;
    dpMax_0->differential_0 = _S37;
    return;
}

__device__ float clamp_0(float x_4, float minBound_0, float maxBound_0)
{
    return (F32_min(((F32_max((x_4), (minBound_0)))), (maxBound_0)));
}

__device__ bool particleDensityHitInterval(float3  rayOrigin_1, float3  rayDirection_1, gaussianParticle_Parameters_0 parameters_1, float minHitDistance_0, float maxHitDistance_0, float * alpha_1, float * depth_1, float3  * canonicalIntersection_1, bool enableNormal_1, float3  * normal_1)
{
    float3  canonicalRayOrigin_1 = mul_0(parameters_1.inverseFactor_0, rayOrigin_1 - parameters_1.position_0);
    float3  canonicalRayDirection_1 = mul_0(parameters_1.inverseFactor_0, rayDirection_1);
    float _S40 = clamp_0(- dot_0(canonicalRayOrigin_1, canonicalRayDirection_1) / dot_0(canonicalRayDirection_1, canonicalRayDirection_1), minHitDistance_0, maxHitDistance_0);
    *depth_1 = _S40;
    float3  _S41 = canonicalRayOrigin_1 + make_float3 (_S40) * canonicalRayDirection_1;
    *canonicalIntersection_1 = _S41;
    float _S42 = (F32_exp((-0.5f * dot_0(_S41, _S41))));
    *alpha_1 = (F32_min((0.99000000953674316f), (_S42 * parameters_1.density_0)));
    bool acceptHit_1;
    if(maxHitDistance_0 > minHitDistance_0)
    {
        acceptHit_1 = _S42 > 0.01130000036209822f;
    }
    else
    {
        acceptHit_1 = false;
    }
    if(acceptHit_1)
    {
        acceptHit_1 = (*alpha_1) > 0.00392156885936856f;
    }
    else
    {
        acceptHit_1 = false;
    }
    bool _S43;
    if(acceptHit_1)
    {
        _S43 = enableNormal_1;
    }
    else
    {
        _S43 = false;
    }
    if(_S43)
    {
        *normal_1 = - normalize_0(mul_0(transpose_0(parameters_1.inverseFactor_0), normalize_0(canonicalRayDirection_1)));
    }
    return acceptHit_1;
}

__device__ void _d_lerp_0(DiffPair_float_0 * dpx_6, DiffPair_float_0 * dpy_3, DiffPair_float_0 * dps_0, float dOut_8)
{
    float _S44 = (1.0f - (*dps_0).primal_0) * dOut_8;
    dpx_6->primal_0 = (*dpx_6).primal_0;
    dpx_6->differential_0 = _S44;
    DiffPair_float_0 _S45 = *dpy_3;
    float _S46 = (*dps_0).primal_0 * dOut_8;
    dpy_3->primal_0 = (*dpy_3).primal_0;
    dpy_3->differential_0 = _S46;
    float _S47 = (_S45.primal_0 - (*dpx_6).primal_0) * dOut_8;
    dps_0->primal_0 = _S45.primal_0;
    dps_0->differential_0 = _S47;
    return;
}

__device__ void _d_lerp_vector_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_7, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_4, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpz_0, float3  dOut_9)
{
    DiffPair_float_0 left_dp_0;
    (&left_dp_0)->primal_0 = (*dpx_7).primal_0.x;
    (&left_dp_0)->differential_0 = 0.0f;
    DiffPair_float_0 middle_dp_0;
    (&middle_dp_0)->primal_0 = (*dpy_4).primal_0.x;
    (&middle_dp_0)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_0;
    (&right_dp_0)->primal_0 = (*dpz_0).primal_0.x;
    (&right_dp_0)->differential_0 = 0.0f;
    _d_lerp_0(&left_dp_0, &middle_dp_0, &right_dp_0, dOut_9.x);
    float3  left_d_result_1;
    *&((&left_d_result_1)->x) = left_dp_0.differential_0;
    float3  middle_d_result_0;
    *&((&middle_d_result_0)->x) = middle_dp_0.differential_0;
    float3  right_d_result_1;
    *&((&right_d_result_1)->x) = right_dp_0.differential_0;
    DiffPair_float_0 left_dp_1;
    (&left_dp_1)->primal_0 = (*dpx_7).primal_0.y;
    (&left_dp_1)->differential_0 = 0.0f;
    DiffPair_float_0 middle_dp_1;
    (&middle_dp_1)->primal_0 = (*dpy_4).primal_0.y;
    (&middle_dp_1)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_1;
    (&right_dp_1)->primal_0 = (*dpz_0).primal_0.y;
    (&right_dp_1)->differential_0 = 0.0f;
    _d_lerp_0(&left_dp_1, &middle_dp_1, &right_dp_1, dOut_9.y);
    *&((&left_d_result_1)->y) = left_dp_1.differential_0;
    *&((&middle_d_result_0)->y) = middle_dp_1.differential_0;
    *&((&right_d_result_1)->y) = right_dp_1.differential_0;
    DiffPair_float_0 left_dp_2;
    (&left_dp_2)->primal_0 = (*dpx_7).primal_0.z;
    (&left_dp_2)->differential_0 = 0.0f;
    DiffPair_float_0 middle_dp_2;
    (&middle_dp_2)->primal_0 = (*dpy_4).primal_0.z;
    (&middle_dp_2)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_2;
    (&right_dp_2)->primal_0 = (*dpz_0).primal_0.z;
    (&right_dp_2)->differential_0 = 0.0f;
    _d_lerp_0(&left_dp_2, &middle_dp_2, &right_dp_2, dOut_9.z);
    *&((&left_d_result_1)->z) = left_dp_2.differential_0;
    *&((&middle_d_result_0)->z) = middle_dp_2.differential_0;
    *&((&right_d_result_1)->z) = right_dp_2.differential_0;
    dpx_7->primal_0 = (*dpx_7).primal_0;
    dpx_7->differential_0 = left_d_result_1;
    dpy_4->primal_0 = (*dpy_4).primal_0;
    dpy_4->differential_0 = middle_d_result_0;
    dpz_0->primal_0 = (*dpz_0).primal_0;
    dpz_0->differential_0 = right_d_result_1;
    return;
}

__device__ float particleDensityIntegrateHit(float alpha_2, float * transmittance_0, float depth_2, float * integratedDepth_0, bool enableNormal_2, float3  normal_2, float3  * integratedNormal_0)
{
    float _S48 = alpha_2 * *transmittance_0;
    *integratedDepth_0 = *integratedDepth_0 + depth_2 * _S48;
    if(enableNormal_2)
    {
        *integratedNormal_0 = *integratedNormal_0 + normal_2 * make_float3 (_S48);
    }
    *transmittance_0 = *transmittance_0 * (1.0f - alpha_2);
    return _S48;
}

__device__ float particleDensityProcessHitFwdFromBuffer(float3  rayOrigin_2, float3  rayDirection_2, uint particleIdx_1, gaussianParticle_CommonParameters_0 commonParameters_1, float * transmittance_1, float * integratedDepth_1, float3  * canonicalIntersection_2, bool enableNormal_3, float3  * integratedNormal_1)
{
    float depth_3;
    for(;;)
    {
        gaussianParticle_RawParameters_0 * _S49 = commonParameters_1.parametersBuffer_0._dataPtr_0 + particleIdx_1;
        gaussianParticle_Parameters_0 _S50 = gaussianParticle_Parameters_x24init_0((*_S49).position_1, (*_S49).factor_1, (*_S49).inverseFactor_1, (*_S49).density_1);
        float3  canonicalRayOrigin_2 = mul_0(_S50.inverseFactor_0, rayOrigin_2 - _S50.position_0);
        float3  grdu_1 = mul_0(_S50.inverseFactor_0, rayDirection_2);
        float3  canonicalRayDirection_2 = normalize_0(grdu_1);
        float3  _S51 = cross_0(canonicalRayDirection_2, canonicalRayOrigin_2);
        float _S52 = (F32_exp((-0.5f * dot_0(_S51, _S51))));
        float alpha_3 = (F32_min((0.99000000953674316f), (_S52 * _S50.density_0)));
        bool acceptHit_2;
        if(_S52 > 0.01130000036209822f)
        {
            acceptHit_2 = alpha_3 > 0.00392156885936856f;
        }
        else
        {
            acceptHit_2 = false;
        }
        float3  normal_3;
        if(acceptHit_2)
        {
            float depth_4 = - dot_0(canonicalRayOrigin_2, grdu_1) / dot_0(grdu_1, grdu_1);
            *canonicalIntersection_2 = canonicalRayOrigin_2 + make_float3 (depth_4) * grdu_1;
            if(enableNormal_3)
            {
                normal_3 = - normalize_0(mul_0(transpose_0(_S50.inverseFactor_0), canonicalRayDirection_2));
            }
            depth_3 = depth_4;
        }
        if(acceptHit_2)
        {
            float _S53 = alpha_3 * *transmittance_1;
            *integratedDepth_1 = *integratedDepth_1 + depth_3 * _S53;
            if(enableNormal_3)
            {
                *integratedNormal_1 = *integratedNormal_1 + normal_3 * make_float3 (_S53);
            }
            *transmittance_1 = *transmittance_1 * (1.0f - alpha_3);
            depth_3 = _S53;
            break;
        }
        depth_3 = 0.0f;
        break;
    }
    return depth_3;
}

__device__ float particleDensityProcessHitFwdFromBufferInterval(float3  rayOrigin_3, float3  rayDirection_3, uint particleIdx_2, gaussianParticle_CommonParameters_0 commonParameters_2, float minHitDistance_1, float maxHitDistance_1, float * transmittance_2, float * integratedDepth_2, float3  * canonicalIntersection_3, bool enableNormal_4, float3  * integratedNormal_2)
{
    float _S54;
    for(;;)
    {
        float3  _S55 = make_float3 (0.0f);
        gaussianParticle_RawParameters_0 * _S56 = commonParameters_2.parametersBuffer_0._dataPtr_0 + particleIdx_2;
        gaussianParticle_Parameters_0 _S57 = gaussianParticle_Parameters_x24init_0((*_S56).position_1, (*_S56).factor_1, (*_S56).inverseFactor_1, (*_S56).density_1);
        float3  canonicalRayOrigin_3 = mul_0(_S57.inverseFactor_0, rayOrigin_3 - _S57.position_0);
        float3  canonicalRayDirection_3 = mul_0(_S57.inverseFactor_0, rayDirection_3);
        float depth_5 = clamp_0(- dot_0(canonicalRayOrigin_3, canonicalRayDirection_3) / dot_0(canonicalRayDirection_3, canonicalRayDirection_3), minHitDistance_1, maxHitDistance_1);
        float3  _S58 = canonicalRayOrigin_3 + make_float3 (depth_5) * canonicalRayDirection_3;
        *canonicalIntersection_3 = _S58;
        float _S59 = (F32_exp((-0.5f * dot_0(_S58, _S58))));
        float alpha_4 = (F32_min((0.99000000953674316f), (_S59 * _S57.density_0)));
        bool acceptHit_3;
        if(maxHitDistance_1 > minHitDistance_1)
        {
            acceptHit_3 = _S59 > 0.01130000036209822f;
        }
        else
        {
            acceptHit_3 = false;
        }
        if(acceptHit_3)
        {
            acceptHit_3 = alpha_4 > 0.00392156885936856f;
        }
        else
        {
            acceptHit_3 = false;
        }
        bool _S60;
        if(acceptHit_3)
        {
            _S60 = enableNormal_4;
        }
        else
        {
            _S60 = false;
        }
        float3  normal_4;
        if(_S60)
        {
            normal_4 = - normalize_0(mul_0(transpose_0(_S57.inverseFactor_0), normalize_0(canonicalRayDirection_3)));
        }
        else
        {
            normal_4 = _S55;
        }
        if(acceptHit_3)
        {
            float _S61 = alpha_4 * *transmittance_2;
            *integratedDepth_2 = *integratedDepth_2 + depth_5 * _S61;
            if(enableNormal_4)
            {
                *integratedNormal_2 = *integratedNormal_2 + normal_4 * make_float3 (_S61);
            }
            *transmittance_2 = *transmittance_2 * (1.0f - alpha_4);
            _S54 = _S61;
            break;
        }
        _S54 = 0.0f;
        break;
    }
    return _S54;
}

struct s_bwd_prop_gaussianParticle_processHitFromBuffer_Intermediates_0
{
    float _S62;
    float _S63;
    float3  _S64;
    gaussianParticle_RawParameters_0 _S65;
};

__device__ gaussianParticle_Parameters_0 s_primal_ctx_gaussianParticle_Parameters_x24init_0(float3  dpposition_0, Matrix<float, 3, 3>  dpfactor_0, Matrix<float, 3, 3>  dpinverseFactor_0, float dpdensity_0)
{
    gaussianParticle_Parameters_0 _S66 = { dpposition_0, dpfactor_0, dpinverseFactor_0, dpdensity_0 };
    return _S66;
}

__device__ float3  s_primal_ctx_mul_0(Matrix<float, 3, 3>  _S67, float3  _S68)
{
    return mul_0(_S67, _S68);
}

__device__ float3  s_primal_ctx_cross_0(float3  _S69, float3  _S70)
{
    return cross_0(_S69, _S70);
}

__device__ float s_primal_ctx_dot_0(float3  _S71, float3  _S72)
{
    return dot_0(_S71, _S72);
}

__device__ float s_primal_ctx_exp_0(float _S73)
{
    return (F32_exp((_S73)));
}

__device__ void s_bwd_prop_lerp_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S74, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S75, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S76, float3  _S77)
{
    _d_lerp_vector_0(_S74, _S75, _S76, _S77);
    return;
}

__device__ void s_bwd_prop_lerp_1(DiffPair_float_0 * _S78, DiffPair_float_0 * _S79, DiffPair_float_0 * _S80, float _S81)
{
    _d_lerp_0(_S78, _S79, _S80, _S81);
    return;
}

__device__ void s_bwd_prop_sqrt_0(DiffPair_float_0 * _S82, float _S83)
{
    _d_sqrt_0(_S82, _S83);
    return;
}

__device__ void s_bwd_prop_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_8, float _s_dOut_0)
{
    float _S84 = (*dpx_8).primal_0.x;
    float _S85 = (*dpx_8).primal_0.y;
    float _S86 = (*dpx_8).primal_0.z;
    DiffPair_float_0 _S87;
    (&_S87)->primal_0 = _S84 * _S84 + _S85 * _S85 + _S86 * _S86;
    (&_S87)->differential_0 = 0.0f;
    s_bwd_prop_sqrt_0(&_S87, _s_dOut_0);
    float _S88 = (*dpx_8).primal_0.z * _S87.differential_0;
    float _S89 = _S88 + _S88;
    float _S90 = (*dpx_8).primal_0.y * _S87.differential_0;
    float _S91 = _S90 + _S90;
    float _S92 = (*dpx_8).primal_0.x * _S87.differential_0;
    float _S93 = _S92 + _S92;
    float3  _S94 = make_float3 (0.0f);
    *&((&_S94)->z) = _S89;
    *&((&_S94)->y) = _S91;
    *&((&_S94)->x) = _S93;
    dpx_8->primal_0 = (*dpx_8).primal_0;
    dpx_8->differential_0 = _S94;
    return;
}

__device__ void s_bwd_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S95, float _S96)
{
    s_bwd_prop_length_impl_0(_S95, _S96);
    return;
}

__device__ void s_bwd_prop_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_9, float3  _s_dOut_1)
{
    float _S97 = length_0((*dpx_9).primal_0);
    float3  _S98 = (*dpx_9).primal_0 * _s_dOut_1;
    float3  _S99 = make_float3 (1.0f / _S97) * _s_dOut_1;
    float _S100 = - ((_S98.x + _S98.y + _S98.z) / (_S97 * _S97));
    float3  _S101 = make_float3 (0.0f);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S102;
    (&_S102)->primal_0 = (*dpx_9).primal_0;
    (&_S102)->differential_0 = _S101;
    s_bwd_length_impl_0(&_S102, _S100);
    float3  _S103 = _S99 + _S102.differential_0;
    dpx_9->primal_0 = (*dpx_9).primal_0;
    dpx_9->differential_0 = _S103;
    return;
}

__device__ void s_bwd_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S104, float3  _S105)
{
    s_bwd_prop_normalize_impl_0(_S104, _S105);
    return;
}

__device__ void s_bwd_prop_mul_0(DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * _S106, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S107, float3  _S108)
{
    _d_mul_0(_S106, _S107, _S108);
    return;
}

__device__ void s_bwd_prop_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S109, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S110, float _S111)
{
    _d_dot_0(_S109, _S110, _S111);
    return;
}

__device__ void s_bwd_prop_exp_0(DiffPair_float_0 * _S112, float _S113)
{
    _d_exp_0(_S112, _S113);
    return;
}

__device__ void s_bwd_prop_cross_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S114, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S115, float3  _S116)
{
    _d_cross_0(_S114, _S115, _S116);
    return;
}

__device__ void s_bwd_prop_gaussianParticle_Parameters_x24init_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpposition_1, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * dpfactor_1, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * dpinverseFactor_1, DiffPair_float_0 * dpdensity_1, gaussianParticle_Parameters_0 * _s_dOut_2)
{
    float3  _S117 = _s_dOut_2->position_0;
    Matrix<float, 3, 3>  _S118 = _s_dOut_2->factor_0;
    Matrix<float, 3, 3>  _S119 = _s_dOut_2->inverseFactor_0;
    float _S120 = _s_dOut_2->density_0;
    dpdensity_1->primal_0 = (*dpdensity_1).primal_0;
    dpdensity_1->differential_0 = _S120;
    dpinverseFactor_1->primal_0 = (*dpinverseFactor_1).primal_0;
    dpinverseFactor_1->differential_0 = _S119;
    dpfactor_1->primal_0 = (*dpfactor_1).primal_0;
    dpfactor_1->differential_0 = _S118;
    dpposition_1->primal_0 = (*dpposition_1).primal_0;
    dpposition_1->differential_0 = _S117;
    return;
}

__device__ void particleDensityProcessHitBwdToBuffer(float3  rayOrigin_4, float3  rayDirection_4, uint particleIdx_3, gaussianParticle_CommonParameters_0 commonParameters_3, float alpha_5, float alphaGrad_0, float * transmittance_3, float * transmittanceGrad_0, float depth_6, float * integratedDepth_3, float * integratedDepthGrad_0, float3  canonicalIntersectionGrad_0, bool enableNormal_5, float3  normal_5, float3  * integratedNormal_3, float3  * integratedNormalGrad_0)
{
    if(alpha_5 > 0.0f)
    {
        float weight_0 = 1.0f / (1.0f - alpha_5);
        float _S121 = *transmittance_3 * weight_0;
        *transmittance_3 = _S121;
        float _S122 = *transmittanceGrad_0;
        float _S123 = (*integratedDepth_3 - depth_6 * alpha_5) * weight_0;
        *integratedDepth_3 = _S123;
        float _S124 = *integratedDepthGrad_0;
        DiffPair_vectorx3Cfloatx2C3x3E_0 integratedNormalDiff_0;
        if(enableNormal_5)
        {
            float3  _S125 = (*integratedNormal_3 - normal_5 * make_float3 (alpha_5)) * make_float3 (weight_0);
            *integratedNormal_3 = _S125;
            (&integratedNormalDiff_0)->primal_0 = _S125;
            (&integratedNormalDiff_0)->differential_0 = *integratedNormalGrad_0;
        }
        else
        {
            int3  _S126 = make_int3 (int(0));
            float3  _S127 = make_float3 ((float)_S126.x, (float)_S126.y, (float)_S126.z);
            (&integratedNormalDiff_0)->primal_0 = _S127;
            (&integratedNormalDiff_0)->differential_0 = _S127;
        }
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S128 = integratedNormalDiff_0;
        s_bwd_prop_gaussianParticle_processHitFromBuffer_Intermediates_0 _S129;
        float3  _S130 = make_float3 (0.0f);
        float4  _S131 = make_float4 (0.0f);
        Matrix<float, 3, 3>  _S132 = makeMatrix<float, 3, 3> (0.0f);
        float2  _S133 = make_float2 (0.0f);
        gaussianParticle_RawParameters_0 _S134 = { _S130, 0.0f, _S131, _S130, 0.0f, _S132, _S132, _S133 };
        (&_S129)->_S62 = 0.0f;
        (&_S129)->_S63 = 0.0f;
        (&_S129)->_S64 = _S130;
        (&_S129)->_S65 = _S134;
        (&(&_S129)->_S65)->position_1 = _S130;
        (&(&_S129)->_S65)->density_1 = 0.0f;
        (&(&_S129)->_S65)->quaternion_0 = _S131;
        (&(&_S129)->_S65)->scale_0 = _S130;
        (&(&_S129)->_S65)->padding_0 = 0.0f;
        (&(&_S129)->_S65)->factor_1 = _S132;
        (&(&_S129)->_S65)->inverseFactor_1 = _S132;
        (&(&_S129)->_S65)->fullShapePadding_0 = _S133;
        (&_S129)->_S62 = _S121;
        (&_S129)->_S63 = _S123;
        (&_S129)->_S64 = _S128.primal_0;
        (&_S129)->_S65 = *(commonParameters_3.parametersBuffer_0._dataPtr_0 + particleIdx_3);
        s_bwd_prop_gaussianParticle_processHitFromBuffer_Intermediates_0 _S135 = _S129;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S136 = integratedNormalDiff_0;
        gaussianParticle_Parameters_0 _S137 = s_primal_ctx_gaussianParticle_Parameters_x24init_0(_S129._S65.position_1, _S129._S65.factor_1, _S129._S65.inverseFactor_1, _S129._S65.density_1);
        float3  gposc_0 = rayOrigin_4 - _S137.position_0;
        float3  _S138 = s_primal_ctx_mul_0(_S137.inverseFactor_0, gposc_0);
        float3  _S139 = s_primal_ctx_mul_0(_S137.inverseFactor_0, rayDirection_4);
        float3  _S140 = normalize_0(_S139);
        float3  _S141 = s_primal_ctx_cross_0(_S140, _S138);
        float _S142 = -0.5f * s_primal_ctx_dot_0(_S141, _S141);
        float _S143 = s_primal_ctx_exp_0(_S142);
        float _S144 = _S143 * _S137.density_0;
        float _S145 = (F32_min((0.99000000953674316f), (_S144)));
        bool acceptHit_4;
        if(_S143 > 0.01130000036209822f)
        {
            acceptHit_4 = _S145 > 0.00392156885936856f;
        }
        else
        {
            acceptHit_4 = false;
        }
        float3  normal_6;
        float3  _S146;
        float3  _S147;
        Matrix<float, 3, 3>  _S148;
        float depth_7;
        float _S149;
        float _S150;
        float _S151;
        if(acceptHit_4)
        {
            float _S152 = - s_primal_ctx_dot_0(_S138, _S139);
            float _S153 = s_primal_ctx_dot_0(_S139, _S139);
            float depth_8 = _S152 / _S153;
            float3  _S154 = make_float3 (depth_8);
            float _S155 = _S153 * _S153;
            if(enableNormal_5)
            {
                Matrix<float, 3, 3>  _S156 = transpose_0(_S137.inverseFactor_0);
                float3  _S157 = s_primal_ctx_mul_0(_S156, _S140);
                normal_6 = - normalize_0(_S157);
                _S146 = _S157;
                _S148 = _S156;
            }
            else
            {
                _S146 = _S130;
                _S148 = _S132;
            }
            depth_7 = depth_8;
            _S147 = _S154;
            _S149 = _S155;
            _S150 = _S152;
            _S151 = _S153;
        }
        else
        {
            _S146 = _S130;
            _S148 = _S132;
            _S147 = _S130;
            _S149 = 0.0f;
            _S150 = 0.0f;
            _S151 = 0.0f;
        }
        bool _runFlag_0;
        float3  dpintegratedNormal_0;
        float _S158;
        if(acceptHit_4)
        {
            if(enableNormal_5)
            {
                dpintegratedNormal_0 = make_float3 (_S145);
            }
            else
            {
                dpintegratedNormal_0 = _S130;
            }
            float _S159 = 1.0f - _S145;
            _runFlag_0 = false;
            _S158 = _S159;
        }
        else
        {
            _runFlag_0 = true;
            _S158 = 0.0f;
            dpintegratedNormal_0 = _S130;
        }
        float dptransmittance_0;
        if(_runFlag_0)
        {
            dptransmittance_0 = 0.0f;
        }
        else
        {
            dptransmittance_0 = alphaGrad_0;
        }
        float dpintegratedDepth_0;
        if(acceptHit_4)
        {
            float _S160 = _S158 * _S122;
            float _S161 = - (_S135._S62 * _S122) + dptransmittance_0;
            if(enableNormal_5)
            {
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S162;
                (&_S162)->primal_0 = _S135._S64;
                (&_S162)->differential_0 = _S130;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S163;
                (&_S163)->primal_0 = normal_6;
                (&_S163)->differential_0 = _S130;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S164;
                (&_S164)->primal_0 = dpintegratedNormal_0;
                (&_S164)->differential_0 = _S130;
                s_bwd_prop_lerp_0(&_S162, &_S163, &_S164, _S136.differential_0);
                _S158 = _S164.differential_0.x + _S164.differential_0.y + _S164.differential_0.z + _S161;
                normal_6 = _S163.differential_0;
                dpintegratedNormal_0 = _S162.differential_0;
            }
            else
            {
                _S158 = _S161;
                normal_6 = _S130;
                dpintegratedNormal_0 = _S136.differential_0;
            }
            DiffPair_float_0 _S165;
            (&_S165)->primal_0 = _S135._S63;
            (&_S165)->differential_0 = 0.0f;
            DiffPair_float_0 _S166;
            (&_S166)->primal_0 = depth_7;
            (&_S166)->differential_0 = 0.0f;
            DiffPair_float_0 _S167;
            (&_S167)->primal_0 = _S145;
            (&_S167)->differential_0 = 0.0f;
            s_bwd_prop_lerp_1(&_S165, &_S166, &_S167, _S124);
            float _S168 = _S167.differential_0 + _S158;
            depth_7 = _S166.differential_0;
            _S158 = _S168;
            dptransmittance_0 = _S160;
            dpintegratedDepth_0 = _S165.differential_0;
        }
        else
        {
            normal_6 = _S130;
            depth_7 = 0.0f;
            _S158 = dptransmittance_0;
            dptransmittance_0 = _S122;
            dpintegratedDepth_0 = _S124;
            dpintegratedNormal_0 = _S136.differential_0;
        }
        if(acceptHit_4)
        {
            if(enableNormal_5)
            {
                float3  _S169 = - normal_6;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S170;
                (&_S170)->primal_0 = _S146;
                (&_S170)->differential_0 = _S130;
                s_bwd_normalize_impl_0(&_S170, _S169);
                DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S171;
                (&_S171)->primal_0 = _S148;
                (&_S171)->differential_0 = _S132;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S172;
                (&_S172)->primal_0 = _S140;
                (&_S172)->differential_0 = _S130;
                s_bwd_prop_mul_0(&_S171, &_S172, _S170.differential_0);
                Matrix<float, 3, 3>  _S173 = transpose_0(_S171.differential_0);
                normal_6 = _S172.differential_0;
                _S148 = _S173;
            }
            else
            {
                normal_6 = _S130;
                _S148 = _S132;
            }
            float3  _S174 = _S147 * canonicalIntersectionGrad_0;
            float3  _S175 = _S139 * canonicalIntersectionGrad_0;
            float _S176 = (_S175.x + _S175.y + _S175.z + depth_7) / _S149;
            float _S177 = _S150 * - _S176;
            float _S178 = _S151 * _S176;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S179;
            (&_S179)->primal_0 = _S139;
            (&_S179)->differential_0 = _S130;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S180;
            (&_S180)->primal_0 = _S139;
            (&_S180)->differential_0 = _S130;
            s_bwd_prop_dot_0(&_S179, &_S180, _S177);
            float _S181 = - _S178;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S182;
            (&_S182)->primal_0 = _S138;
            (&_S182)->differential_0 = _S130;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S183;
            (&_S183)->primal_0 = _S139;
            (&_S183)->differential_0 = _S130;
            s_bwd_prop_dot_0(&_S182, &_S183, _S181);
            float3  _S184 = canonicalIntersectionGrad_0 + _S182.differential_0;
            _S146 = _S174 + _S180.differential_0 + _S179.differential_0 + _S183.differential_0;
            _S147 = _S184;
        }
        else
        {
            normal_6 = _S130;
            _S146 = _S130;
            _S147 = _S130;
            _S148 = _S132;
        }
        DiffPair_float_0 _S185;
        (&_S185)->primal_0 = 0.99000000953674316f;
        (&_S185)->differential_0 = 0.0f;
        DiffPair_float_0 _S186;
        (&_S186)->primal_0 = _S144;
        (&_S186)->differential_0 = 0.0f;
        _d_min_0(&_S185, &_S186, _S158);
        float _S187 = _S143 * _S186.differential_0;
        float _S188 = _S137.density_0 * _S186.differential_0;
        DiffPair_float_0 _S189;
        (&_S189)->primal_0 = _S142;
        (&_S189)->differential_0 = 0.0f;
        s_bwd_prop_exp_0(&_S189, _S188);
        float _S190 = -0.5f * _S189.differential_0;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S191;
        (&_S191)->primal_0 = _S141;
        (&_S191)->differential_0 = _S130;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S192;
        (&_S192)->primal_0 = _S141;
        (&_S192)->differential_0 = _S130;
        s_bwd_prop_dot_0(&_S191, &_S192, _S190);
        float3  _S193 = _S192.differential_0 + _S191.differential_0;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S194;
        (&_S194)->primal_0 = _S140;
        (&_S194)->differential_0 = _S130;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S195;
        (&_S195)->primal_0 = _S138;
        (&_S195)->differential_0 = _S130;
        s_bwd_prop_cross_0(&_S194, &_S195, _S193);
        float3  _S196 = _S194.differential_0 + normal_6;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S197;
        (&_S197)->primal_0 = _S139;
        (&_S197)->differential_0 = _S130;
        s_bwd_normalize_impl_0(&_S197, _S196);
        float3  _S198 = _S197.differential_0 + _S146;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S199;
        (&_S199)->primal_0 = _S137.inverseFactor_0;
        (&_S199)->differential_0 = _S132;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S200;
        (&_S200)->primal_0 = rayDirection_4;
        (&_S200)->differential_0 = _S130;
        s_bwd_prop_mul_0(&_S199, &_S200, _S198);
        float3  _S201 = _S195.differential_0 + _S147;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S202;
        (&_S202)->primal_0 = _S137.inverseFactor_0;
        (&_S202)->differential_0 = _S132;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S203;
        (&_S203)->primal_0 = gposc_0;
        (&_S203)->differential_0 = _S130;
        s_bwd_prop_mul_0(&_S202, &_S203, _S201);
        Matrix<float, 3, 3>  _S204 = _S199.differential_0 + _S202.differential_0 + _S148;
        float3  _S205 = - _S203.differential_0;
        gaussianParticle_Parameters_0 _S206 = gaussianParticle_Parameters_x24_syn_dzero_0();
        (&_S206)->density_0 = _S187;
        (&_S206)->inverseFactor_0 = _S204;
        (&_S206)->position_0 = _S205;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S207;
        (&_S207)->primal_0 = _S135._S65.position_1;
        (&_S207)->differential_0 = _S130;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S208;
        (&_S208)->primal_0 = _S135._S65.factor_1;
        (&_S208)->differential_0 = _S132;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S209;
        (&_S209)->primal_0 = _S135._S65.inverseFactor_1;
        (&_S209)->differential_0 = _S132;
        DiffPair_float_0 _S210;
        (&_S210)->primal_0 = _S135._S65.density_1;
        (&_S210)->differential_0 = 0.0f;
        gaussianParticle_Parameters_0 _S211 = _S206;
        s_bwd_prop_gaussianParticle_Parameters_x24init_0(&_S207, &_S208, &_S209, &_S210, &_S211);
        gaussianParticle_RawParameters_0 _S212 = gaussianParticle_RawParameters_x24_syn_dzero_0();
        (&_S212)->density_1 = _S210.differential_0;
        (&_S212)->inverseFactor_1 = _S209.differential_0;
        (&_S212)->factor_1 = _S208.differential_0;
        (&_S212)->position_1 = _S207.differential_0;
        gaussianParticle_RawParameters_0 _S213 = _S212;
        if(commonParameters_3.parametersBuffer_0.exclusiveGradient_0)
        {
            gaussianParticle_RawParameters_0 * _S214 = commonParameters_3.parametersBuffer_0._gradPtr_0 + particleIdx_3;
            _S214->density_1 = _S214->density_1 + _S213.density_1;
            _S214->position_1 = _S214->position_1 + _S213.position_1;
            _S214->quaternion_0 = _S214->quaternion_0 + _S213.quaternion_0;
            _S214->scale_0 = _S214->scale_0 + _S213.scale_0;
            _S214->factor_1 = _S214->factor_1 + _S213.factor_1;
            _S214->inverseFactor_1 = _S214->inverseFactor_1 + _S213.inverseFactor_1;
        }
        else
        {
            gaussianParticle_RawParameters_0 * _S215 = commonParameters_3.parametersBuffer_0._gradPtr_0 + particleIdx_3;
            float _S216 = atomicAdd(&_S215->density_1, _S213.density_1);
            float _S217 = atomicAdd(&((&_S215->position_1)->x), _S213.position_1.x);
            float _S218 = atomicAdd(&((&_S215->position_1)->y), _S213.position_1.y);
            float _S219 = atomicAdd(&((&_S215->position_1)->z), _S213.position_1.z);
            float _S220 = atomicAdd(&((&_S215->quaternion_0)->x), _S213.quaternion_0.x);
            float _S221 = atomicAdd(&((&_S215->quaternion_0)->y), _S213.quaternion_0.y);
            float _S222 = atomicAdd(&((&_S215->quaternion_0)->z), _S213.quaternion_0.z);
            float _S223 = atomicAdd(&((&_S215->quaternion_0)->w), _S213.quaternion_0.w);
            float _S224 = atomicAdd(&((&_S215->scale_0)->x), _S213.scale_0.x);
            float _S225 = atomicAdd(&((&_S215->scale_0)->y), _S213.scale_0.y);
            float _S226 = atomicAdd(&((&_S215->scale_0)->z), _S213.scale_0.z);
            float _S227 = atomicAdd(&(((&_S215->factor_1)->rows + (int(0)))->x), _S213.factor_1.rows[int(0)].x);
            float _S228 = atomicAdd(&(((&_S215->inverseFactor_1)->rows + (int(0)))->x), _S213.inverseFactor_1.rows[int(0)].x);
            float _S229 = atomicAdd(&(((&_S215->factor_1)->rows + (int(0)))->y), _S213.factor_1.rows[int(0)].y);
            float _S230 = atomicAdd(&(((&_S215->inverseFactor_1)->rows + (int(0)))->y), _S213.inverseFactor_1.rows[int(0)].y);
            float _S231 = atomicAdd(&(((&_S215->factor_1)->rows + (int(0)))->z), _S213.factor_1.rows[int(0)].z);
            float _S232 = atomicAdd(&(((&_S215->inverseFactor_1)->rows + (int(0)))->z), _S213.inverseFactor_1.rows[int(0)].z);
            float _S233 = atomicAdd(&(((&_S215->factor_1)->rows + (int(1)))->x), _S213.factor_1.rows[int(1)].x);
            float _S234 = atomicAdd(&(((&_S215->inverseFactor_1)->rows + (int(1)))->x), _S213.inverseFactor_1.rows[int(1)].x);
            float _S235 = atomicAdd(&(((&_S215->factor_1)->rows + (int(1)))->y), _S213.factor_1.rows[int(1)].y);
            float _S236 = atomicAdd(&(((&_S215->inverseFactor_1)->rows + (int(1)))->y), _S213.inverseFactor_1.rows[int(1)].y);
            float _S237 = atomicAdd(&(((&_S215->factor_1)->rows + (int(1)))->z), _S213.factor_1.rows[int(1)].z);
            float _S238 = atomicAdd(&(((&_S215->inverseFactor_1)->rows + (int(1)))->z), _S213.inverseFactor_1.rows[int(1)].z);
            float _S239 = atomicAdd(&(((&_S215->factor_1)->rows + (int(2)))->x), _S213.factor_1.rows[int(2)].x);
            float _S240 = atomicAdd(&(((&_S215->inverseFactor_1)->rows + (int(2)))->x), _S213.inverseFactor_1.rows[int(2)].x);
            float _S241 = atomicAdd(&(((&_S215->factor_1)->rows + (int(2)))->y), _S213.factor_1.rows[int(2)].y);
            float _S242 = atomicAdd(&(((&_S215->inverseFactor_1)->rows + (int(2)))->y), _S213.inverseFactor_1.rows[int(2)].y);
            float _S243 = atomicAdd(&(((&_S215->factor_1)->rows + (int(2)))->z), _S213.factor_1.rows[int(2)].z);
            float _S244 = atomicAdd(&(((&_S215->inverseFactor_1)->rows + (int(2)))->z), _S213.inverseFactor_1.rows[int(2)].z);
        }
        *transmittanceGrad_0 = dptransmittance_0;
        *integratedDepthGrad_0 = dpintegratedDepth_0;
        if(enableNormal_5)
        {
            *integratedNormalGrad_0 = dpintegratedNormal_0;
        }
    }
    return;
}

struct s_bwd_prop_gaussianParticle_processHitFromBufferInterval_Intermediates_0
{
    float _S245;
    float _S246;
    float3  _S247;
    gaussianParticle_RawParameters_0 _S248;
};

__device__ float s_primal_ctx_clamp_0(float _S249, float _S250, float _S251)
{
    return clamp_0(_S249, _S250, _S251);
}

__device__ void s_bwd_prop_clamp_0(DiffPair_float_0 * _S252, DiffPair_float_0 * _S253, DiffPair_float_0 * _S254, float _S255)
{
    _d_clamp_0(_S252, _S253, _S254, _S255);
    return;
}

__device__ void particleDensityProcessHitBwdToBufferInterval(float3  rayOrigin_5, float3  rayDirection_5, uint particleIdx_4, gaussianParticle_CommonParameters_0 commonParameters_4, float minHitDistance_2, float maxHitDistance_2, float alpha_6, float alphaGrad_1, float * transmittance_4, float * transmittanceGrad_1, float depth_9, float * integratedDepth_4, float * integratedDepthGrad_1, float3  canonicalIntersectionGrad_1, bool enableNormal_6, float3  normal_7, float3  * integratedNormal_4, float3  * integratedNormalGrad_1)
{
    if(alpha_6 > 0.0f)
    {
        float weight_1 = 1.0f / (1.0f - alpha_6);
        float _S256 = *transmittance_4 * weight_1;
        *transmittance_4 = _S256;
        float _S257 = *transmittanceGrad_1;
        float _S258 = (*integratedDepth_4 - depth_9 * alpha_6) * weight_1;
        *integratedDepth_4 = _S258;
        float _S259 = *integratedDepthGrad_1;
        DiffPair_vectorx3Cfloatx2C3x3E_0 integratedNormalDiff_1;
        if(enableNormal_6)
        {
            float3  _S260 = (*integratedNormal_4 - normal_7 * make_float3 (alpha_6)) * make_float3 (weight_1);
            *integratedNormal_4 = _S260;
            (&integratedNormalDiff_1)->primal_0 = _S260;
            (&integratedNormalDiff_1)->differential_0 = *integratedNormalGrad_1;
        }
        else
        {
            int3  _S261 = make_int3 (int(0));
            float3  _S262 = make_float3 ((float)_S261.x, (float)_S261.y, (float)_S261.z);
            (&integratedNormalDiff_1)->primal_0 = _S262;
            (&integratedNormalDiff_1)->differential_0 = _S262;
        }
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S263 = integratedNormalDiff_1;
        s_bwd_prop_gaussianParticle_processHitFromBufferInterval_Intermediates_0 _S264;
        float3  _S265 = make_float3 (0.0f);
        float4  _S266 = make_float4 (0.0f);
        Matrix<float, 3, 3>  _S267 = makeMatrix<float, 3, 3> (0.0f);
        float2  _S268 = make_float2 (0.0f);
        gaussianParticle_RawParameters_0 _S269 = { _S265, 0.0f, _S266, _S265, 0.0f, _S267, _S267, _S268 };
        (&_S264)->_S245 = 0.0f;
        (&_S264)->_S246 = 0.0f;
        (&_S264)->_S247 = _S265;
        (&_S264)->_S248 = _S269;
        (&(&_S264)->_S248)->position_1 = _S265;
        (&(&_S264)->_S248)->density_1 = 0.0f;
        (&(&_S264)->_S248)->quaternion_0 = _S266;
        (&(&_S264)->_S248)->scale_0 = _S265;
        (&(&_S264)->_S248)->padding_0 = 0.0f;
        (&(&_S264)->_S248)->factor_1 = _S267;
        (&(&_S264)->_S248)->inverseFactor_1 = _S267;
        (&(&_S264)->_S248)->fullShapePadding_0 = _S268;
        (&_S264)->_S245 = _S256;
        (&_S264)->_S246 = _S258;
        (&_S264)->_S247 = _S263.primal_0;
        (&_S264)->_S248 = *(commonParameters_4.parametersBuffer_0._dataPtr_0 + particleIdx_4);
        s_bwd_prop_gaussianParticle_processHitFromBufferInterval_Intermediates_0 _S270 = _S264;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S271 = integratedNormalDiff_1;
        float3  _S272 = make_float3 (0.0f);
        gaussianParticle_Parameters_0 _S273 = s_primal_ctx_gaussianParticle_Parameters_x24init_0(_S264._S248.position_1, _S264._S248.factor_1, _S264._S248.inverseFactor_1, _S264._S248.density_1);
        float3  _S274 = rayOrigin_5 - _S273.position_0;
        float3  _S275 = s_primal_ctx_mul_0(_S273.inverseFactor_0, _S274);
        float3  _S276 = s_primal_ctx_mul_0(_S273.inverseFactor_0, rayDirection_5);
        float _S277 = s_primal_ctx_dot_0(_S276, _S276);
        float _S278 = - s_primal_ctx_dot_0(_S275, _S276);
        float tauLine_0 = _S278 / _S277;
        float _S279 = _S277 * _S277;
        float _S280 = s_primal_ctx_clamp_0(tauLine_0, minHitDistance_2, maxHitDistance_2);
        float3  _S281 = make_float3 (_S280);
        float3  _S282 = _S275 + make_float3 (_S280) * _S276;
        float _S283 = -0.5f * s_primal_ctx_dot_0(_S282, _S282);
        float _S284 = s_primal_ctx_exp_0(_S283);
        float _S285 = _S284 * _S273.density_0;
        float _S286 = (F32_min((0.99000000953674316f), (_S285)));
        bool acceptHit_5;
        if(maxHitDistance_2 > minHitDistance_2)
        {
            acceptHit_5 = _S284 > 0.01130000036209822f;
        }
        else
        {
            acceptHit_5 = false;
        }
        if(acceptHit_5)
        {
            acceptHit_5 = _S286 > 0.00392156885936856f;
        }
        else
        {
            acceptHit_5 = false;
        }
        bool _S287;
        if(acceptHit_5)
        {
            _S287 = enableNormal_6;
        }
        else
        {
            _S287 = false;
        }
        float3  normal_8;
        float3  _S288;
        float3  _S289;
        Matrix<float, 3, 3>  _S290;
        if(_S287)
        {
            float3  _S291 = normalize_0(_S276);
            Matrix<float, 3, 3>  _S292 = transpose_0(_S273.inverseFactor_0);
            float3  _S293 = s_primal_ctx_mul_0(_S292, _S291);
            normal_8 = - normalize_0(_S293);
            _S288 = _S293;
            _S290 = _S292;
            _S289 = _S291;
        }
        else
        {
            normal_8 = _S272;
            _S288 = _S265;
            _S290 = _S267;
            _S289 = _S265;
        }
        bool _runFlag_1;
        float3  dpintegratedNormal_1;
        float _S294;
        if(acceptHit_5)
        {
            if(enableNormal_6)
            {
                dpintegratedNormal_1 = make_float3 (_S286);
            }
            else
            {
                dpintegratedNormal_1 = _S265;
            }
            float _S295 = 1.0f - _S286;
            _runFlag_1 = false;
            _S294 = _S295;
        }
        else
        {
            _runFlag_1 = true;
            _S294 = 0.0f;
            dpintegratedNormal_1 = _S265;
        }
        float _S296;
        if(_runFlag_1)
        {
            _S296 = 0.0f;
        }
        else
        {
            _S296 = alphaGrad_1;
        }
        float dptransmittance_1;
        float dpintegratedDepth_1;
        if(acceptHit_5)
        {
            float _S297 = _S294 * _S257;
            float _S298 = - (_S270._S245 * _S257) + _S296;
            if(enableNormal_6)
            {
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S299;
                (&_S299)->primal_0 = _S270._S247;
                (&_S299)->differential_0 = _S265;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S300;
                (&_S300)->primal_0 = normal_8;
                (&_S300)->differential_0 = _S265;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S301;
                (&_S301)->primal_0 = dpintegratedNormal_1;
                (&_S301)->differential_0 = _S265;
                s_bwd_prop_lerp_0(&_S299, &_S300, &_S301, _S271.differential_0);
                _S294 = _S301.differential_0.x + _S301.differential_0.y + _S301.differential_0.z + _S298;
                normal_8 = _S300.differential_0;
                dpintegratedNormal_1 = _S299.differential_0;
            }
            else
            {
                _S294 = _S298;
                normal_8 = _S265;
                dpintegratedNormal_1 = _S271.differential_0;
            }
            DiffPair_float_0 _S302;
            (&_S302)->primal_0 = _S270._S246;
            (&_S302)->differential_0 = 0.0f;
            DiffPair_float_0 _S303;
            (&_S303)->primal_0 = _S280;
            (&_S303)->differential_0 = 0.0f;
            DiffPair_float_0 _S304;
            (&_S304)->primal_0 = _S286;
            (&_S304)->differential_0 = 0.0f;
            s_bwd_prop_lerp_1(&_S302, &_S303, &_S304, _S259);
            _S294 = _S304.differential_0 + _S294;
            _S296 = _S303.differential_0;
            dptransmittance_1 = _S297;
            dpintegratedDepth_1 = _S302.differential_0;
        }
        else
        {
            normal_8 = _S265;
            _S294 = _S296;
            _S296 = 0.0f;
            dptransmittance_1 = _S257;
            dpintegratedDepth_1 = _S259;
            dpintegratedNormal_1 = _S271.differential_0;
        }
        if(_S287)
        {
            float3  _S305 = - normal_8;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S306;
            (&_S306)->primal_0 = _S288;
            (&_S306)->differential_0 = _S265;
            s_bwd_normalize_impl_0(&_S306, _S305);
            DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S307;
            (&_S307)->primal_0 = _S290;
            (&_S307)->differential_0 = _S267;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S308;
            (&_S308)->primal_0 = _S289;
            (&_S308)->differential_0 = _S265;
            s_bwd_prop_mul_0(&_S307, &_S308, _S306.differential_0);
            Matrix<float, 3, 3>  _S309 = transpose_0(_S307.differential_0);
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S310;
            (&_S310)->primal_0 = _S276;
            (&_S310)->differential_0 = _S265;
            s_bwd_normalize_impl_0(&_S310, _S308.differential_0);
            normal_8 = _S310.differential_0;
            _S290 = _S309;
        }
        else
        {
            normal_8 = _S265;
            _S290 = _S267;
        }
        DiffPair_float_0 _S311;
        (&_S311)->primal_0 = 0.99000000953674316f;
        (&_S311)->differential_0 = 0.0f;
        DiffPair_float_0 _S312;
        (&_S312)->primal_0 = _S285;
        (&_S312)->differential_0 = 0.0f;
        _d_min_0(&_S311, &_S312, _S294);
        float _S313 = _S284 * _S312.differential_0;
        float _S314 = _S273.density_0 * _S312.differential_0;
        DiffPair_float_0 _S315;
        (&_S315)->primal_0 = _S283;
        (&_S315)->differential_0 = 0.0f;
        s_bwd_prop_exp_0(&_S315, _S314);
        float _S316 = -0.5f * _S315.differential_0;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S317;
        (&_S317)->primal_0 = _S282;
        (&_S317)->differential_0 = _S265;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S318;
        (&_S318)->primal_0 = _S282;
        (&_S318)->differential_0 = _S265;
        s_bwd_prop_dot_0(&_S317, &_S318, _S316);
        float3  _S319 = _S318.differential_0 + _S317.differential_0 + canonicalIntersectionGrad_1;
        float3  _S320 = _S281 * _S319;
        float3  _S321 = _S276 * _S319;
        float _S322 = _S321.x + _S321.y + _S321.z + _S296;
        DiffPair_float_0 _S323;
        (&_S323)->primal_0 = tauLine_0;
        (&_S323)->differential_0 = 0.0f;
        DiffPair_float_0 _S324;
        (&_S324)->primal_0 = minHitDistance_2;
        (&_S324)->differential_0 = 0.0f;
        DiffPair_float_0 _S325;
        (&_S325)->primal_0 = maxHitDistance_2;
        (&_S325)->differential_0 = 0.0f;
        s_bwd_prop_clamp_0(&_S323, &_S324, &_S325, _S322);
        float _S326 = _S323.differential_0 / _S279;
        float _S327 = _S278 * - _S326;
        float _S328 = - (_S277 * _S326);
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S329;
        (&_S329)->primal_0 = _S275;
        (&_S329)->differential_0 = _S265;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S330;
        (&_S330)->primal_0 = _S276;
        (&_S330)->differential_0 = _S265;
        s_bwd_prop_dot_0(&_S329, &_S330, _S328);
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S331;
        (&_S331)->primal_0 = _S276;
        (&_S331)->differential_0 = _S265;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S332;
        (&_S332)->primal_0 = _S276;
        (&_S332)->differential_0 = _S265;
        s_bwd_prop_dot_0(&_S331, &_S332, _S327);
        float3  _S333 = _S320 + _S330.differential_0 + _S332.differential_0 + _S331.differential_0 + normal_8;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S334;
        (&_S334)->primal_0 = _S273.inverseFactor_0;
        (&_S334)->differential_0 = _S267;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S335;
        (&_S335)->primal_0 = rayDirection_5;
        (&_S335)->differential_0 = _S265;
        s_bwd_prop_mul_0(&_S334, &_S335, _S333);
        float3  _S336 = _S319 + _S329.differential_0;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S337;
        (&_S337)->primal_0 = _S273.inverseFactor_0;
        (&_S337)->differential_0 = _S267;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S338;
        (&_S338)->primal_0 = _S274;
        (&_S338)->differential_0 = _S265;
        s_bwd_prop_mul_0(&_S337, &_S338, _S336);
        float3  _S339 = - _S338.differential_0;
        Matrix<float, 3, 3>  _S340 = _S334.differential_0 + _S337.differential_0 + _S290;
        gaussianParticle_Parameters_0 _S341 = gaussianParticle_Parameters_x24_syn_dzero_0();
        (&_S341)->density_0 = _S313;
        (&_S341)->position_0 = _S339;
        (&_S341)->inverseFactor_0 = _S340;
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S342;
        (&_S342)->primal_0 = _S270._S248.position_1;
        (&_S342)->differential_0 = _S265;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S343;
        (&_S343)->primal_0 = _S270._S248.factor_1;
        (&_S343)->differential_0 = _S267;
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S344;
        (&_S344)->primal_0 = _S270._S248.inverseFactor_1;
        (&_S344)->differential_0 = _S267;
        DiffPair_float_0 _S345;
        (&_S345)->primal_0 = _S270._S248.density_1;
        (&_S345)->differential_0 = 0.0f;
        gaussianParticle_Parameters_0 _S346 = _S341;
        s_bwd_prop_gaussianParticle_Parameters_x24init_0(&_S342, &_S343, &_S344, &_S345, &_S346);
        gaussianParticle_RawParameters_0 _S347 = gaussianParticle_RawParameters_x24_syn_dzero_0();
        (&_S347)->density_1 = _S345.differential_0;
        (&_S347)->inverseFactor_1 = _S344.differential_0;
        (&_S347)->factor_1 = _S343.differential_0;
        (&_S347)->position_1 = _S342.differential_0;
        gaussianParticle_RawParameters_0 _S348 = _S347;
        if(commonParameters_4.parametersBuffer_0.exclusiveGradient_0)
        {
            gaussianParticle_RawParameters_0 * _S349 = commonParameters_4.parametersBuffer_0._gradPtr_0 + particleIdx_4;
            _S349->density_1 = _S349->density_1 + _S348.density_1;
            _S349->position_1 = _S349->position_1 + _S348.position_1;
            _S349->quaternion_0 = _S349->quaternion_0 + _S348.quaternion_0;
            _S349->scale_0 = _S349->scale_0 + _S348.scale_0;
            _S349->factor_1 = _S349->factor_1 + _S348.factor_1;
            _S349->inverseFactor_1 = _S349->inverseFactor_1 + _S348.inverseFactor_1;
        }
        else
        {
            gaussianParticle_RawParameters_0 * _S350 = commonParameters_4.parametersBuffer_0._gradPtr_0 + particleIdx_4;
            float _S351 = atomicAdd(&_S350->density_1, _S348.density_1);
            float _S352 = atomicAdd(&((&_S350->position_1)->x), _S348.position_1.x);
            float _S353 = atomicAdd(&((&_S350->position_1)->y), _S348.position_1.y);
            float _S354 = atomicAdd(&((&_S350->position_1)->z), _S348.position_1.z);
            float _S355 = atomicAdd(&((&_S350->quaternion_0)->x), _S348.quaternion_0.x);
            float _S356 = atomicAdd(&((&_S350->quaternion_0)->y), _S348.quaternion_0.y);
            float _S357 = atomicAdd(&((&_S350->quaternion_0)->z), _S348.quaternion_0.z);
            float _S358 = atomicAdd(&((&_S350->quaternion_0)->w), _S348.quaternion_0.w);
            float _S359 = atomicAdd(&((&_S350->scale_0)->x), _S348.scale_0.x);
            float _S360 = atomicAdd(&((&_S350->scale_0)->y), _S348.scale_0.y);
            float _S361 = atomicAdd(&((&_S350->scale_0)->z), _S348.scale_0.z);
            float _S362 = atomicAdd(&(((&_S350->factor_1)->rows + (int(0)))->x), _S348.factor_1.rows[int(0)].x);
            float _S363 = atomicAdd(&(((&_S350->inverseFactor_1)->rows + (int(0)))->x), _S348.inverseFactor_1.rows[int(0)].x);
            float _S364 = atomicAdd(&(((&_S350->factor_1)->rows + (int(0)))->y), _S348.factor_1.rows[int(0)].y);
            float _S365 = atomicAdd(&(((&_S350->inverseFactor_1)->rows + (int(0)))->y), _S348.inverseFactor_1.rows[int(0)].y);
            float _S366 = atomicAdd(&(((&_S350->factor_1)->rows + (int(0)))->z), _S348.factor_1.rows[int(0)].z);
            float _S367 = atomicAdd(&(((&_S350->inverseFactor_1)->rows + (int(0)))->z), _S348.inverseFactor_1.rows[int(0)].z);
            float _S368 = atomicAdd(&(((&_S350->factor_1)->rows + (int(1)))->x), _S348.factor_1.rows[int(1)].x);
            float _S369 = atomicAdd(&(((&_S350->inverseFactor_1)->rows + (int(1)))->x), _S348.inverseFactor_1.rows[int(1)].x);
            float _S370 = atomicAdd(&(((&_S350->factor_1)->rows + (int(1)))->y), _S348.factor_1.rows[int(1)].y);
            float _S371 = atomicAdd(&(((&_S350->inverseFactor_1)->rows + (int(1)))->y), _S348.inverseFactor_1.rows[int(1)].y);
            float _S372 = atomicAdd(&(((&_S350->factor_1)->rows + (int(1)))->z), _S348.factor_1.rows[int(1)].z);
            float _S373 = atomicAdd(&(((&_S350->inverseFactor_1)->rows + (int(1)))->z), _S348.inverseFactor_1.rows[int(1)].z);
            float _S374 = atomicAdd(&(((&_S350->factor_1)->rows + (int(2)))->x), _S348.factor_1.rows[int(2)].x);
            float _S375 = atomicAdd(&(((&_S350->inverseFactor_1)->rows + (int(2)))->x), _S348.inverseFactor_1.rows[int(2)].x);
            float _S376 = atomicAdd(&(((&_S350->factor_1)->rows + (int(2)))->y), _S348.factor_1.rows[int(2)].y);
            float _S377 = atomicAdd(&(((&_S350->inverseFactor_1)->rows + (int(2)))->y), _S348.inverseFactor_1.rows[int(2)].y);
            float _S378 = atomicAdd(&(((&_S350->factor_1)->rows + (int(2)))->z), _S348.factor_1.rows[int(2)].z);
            float _S379 = atomicAdd(&(((&_S350->inverseFactor_1)->rows + (int(2)))->z), _S348.inverseFactor_1.rows[int(2)].z);
        }
        *transmittanceGrad_1 = dptransmittance_1;
        *integratedDepthGrad_1 = dpintegratedDepth_1;
        if(enableNormal_6)
        {
            *integratedNormalGrad_1 = dpintegratedNormal_1;
        }
    }
    return;
}

__device__ bool particleDensityHitCustom(float3  rayOrigin_6, float3  rayDirection_6, int particleIdx_5, gaussianParticle_CommonParameters_0 commonParameters_5, float minHitDistance_3, float maxHitDistance_3, float maxParticleSquaredDistance_0, float * hitDistance_0)
{
    gaussianParticle_RawParameters_0 * _S380 = commonParameters_5.parametersBuffer_0._dataPtr_0 + uint(particleIdx_5);
    gaussianParticle_Parameters_0 _S381 = gaussianParticle_Parameters_x24init_0((*_S380).position_1, (*_S380).factor_1, (*_S380).inverseFactor_1, (*_S380).density_1);
    float3  canonicalRayOrigin_4 = mul_0(_S381.inverseFactor_0, rayOrigin_6 - _S381.position_0);
    float3  canonicalRayDirection_4 = mul_0(_S381.inverseFactor_0, rayDirection_6);
    float _S382 = clamp_0(- dot_0(canonicalRayOrigin_4, canonicalRayDirection_4) / dot_0(canonicalRayDirection_4, canonicalRayDirection_4), minHitDistance_3, maxHitDistance_3);
    *hitDistance_0 = _S382;
    float3  canonicalIntersection_4 = canonicalRayOrigin_4 + make_float3 (_S382) * canonicalRayDirection_4;
    bool _S383;
    if(maxHitDistance_3 > minHitDistance_3)
    {
        _S383 = (dot_0(canonicalIntersection_4, canonicalIntersection_4)) < maxParticleSquaredDistance_0;
    }
    else
    {
        _S383 = false;
    }
    return _S383;
}

__device__ bool particleDensityHitInstance(float3  canonicalRayOrigin_5, float3  canonicalUnormalizedRayDirection_0, float minHitDistance_4, float maxHitDistance_4, float maxParticleSquaredDistance_1, float * hitDistance_1)
{
    float _S384 = clamp_0(- dot_0(canonicalRayOrigin_5, canonicalUnormalizedRayDirection_0) / dot_0(canonicalUnormalizedRayDirection_0, canonicalUnormalizedRayDirection_0), minHitDistance_4, maxHitDistance_4);
    *hitDistance_1 = _S384;
    float3  canonicalIntersection_5 = canonicalRayOrigin_5 + make_float3 (_S384) * canonicalUnormalizedRayDirection_0;
    bool _S385;
    if(maxHitDistance_4 > minHitDistance_4)
    {
        _S385 = (dot_0(canonicalIntersection_5, canonicalIntersection_5)) < maxParticleSquaredDistance_1;
    }
    else
    {
        _S385 = false;
    }
    return _S385;
}

__device__ float3  particleDensityIncidentDirection(gaussianParticle_Parameters_0 parameters_2, float3  sourcePosition_0)
{
    return normalize_0(parameters_2.position_0 - sourcePosition_0);
}

struct s_bwd_prop_gaussianParticle_incidentDirectionFromBuffer_Intermediates_0
{
    gaussianParticle_RawParameters_0 _S386;
};

__device__ void particleDensityIncidentDirectionBwdToBuffer(uint particleIdx_6, gaussianParticle_CommonParameters_0 commonParameters_6, float3  sourcePosition_1, float3  incidentDirectionGrad_0)
{
    float3  _S387 = make_float3 (0.0f);
    float4  _S388 = make_float4 (0.0f);
    Matrix<float, 3, 3>  _S389 = makeMatrix<float, 3, 3> (0.0f);
    float2  _S390 = make_float2 (0.0f);
    gaussianParticle_RawParameters_0 _S391 = { _S387, 0.0f, _S388, _S387, 0.0f, _S389, _S389, _S390 };
    s_bwd_prop_gaussianParticle_incidentDirectionFromBuffer_Intermediates_0 _S392;
    (&_S392)->_S386 = _S391;
    (&(&_S392)->_S386)->position_1 = _S387;
    (&(&_S392)->_S386)->density_1 = 0.0f;
    (&(&_S392)->_S386)->quaternion_0 = _S388;
    (&(&_S392)->_S386)->scale_0 = _S387;
    (&(&_S392)->_S386)->padding_0 = 0.0f;
    (&(&_S392)->_S386)->factor_1 = _S389;
    (&(&_S392)->_S386)->inverseFactor_1 = _S389;
    (&(&_S392)->_S386)->fullShapePadding_0 = _S390;
    (&_S392)->_S386 = *(commonParameters_6.parametersBuffer_0._dataPtr_0 + particleIdx_6);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S393;
    (&_S393)->primal_0 = s_primal_ctx_gaussianParticle_Parameters_x24init_0(_S392._S386.position_1, _S392._S386.factor_1, _S392._S386.inverseFactor_1, _S392._S386.density_1).position_0 - sourcePosition_1;
    (&_S393)->differential_0 = _S387;
    s_bwd_normalize_impl_0(&_S393, incidentDirectionGrad_0);
    gaussianParticle_Parameters_0 _S394 = gaussianParticle_Parameters_x24_syn_dzero_0();
    (&_S394)->position_0 = _S393.differential_0;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S395;
    (&_S395)->primal_0 = _S392._S386.position_1;
    (&_S395)->differential_0 = _S387;
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S396;
    (&_S396)->primal_0 = _S392._S386.factor_1;
    (&_S396)->differential_0 = _S389;
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S397;
    (&_S397)->primal_0 = _S392._S386.inverseFactor_1;
    (&_S397)->differential_0 = _S389;
    DiffPair_float_0 _S398;
    (&_S398)->primal_0 = _S392._S386.density_1;
    (&_S398)->differential_0 = 0.0f;
    gaussianParticle_Parameters_0 _S399 = _S394;
    s_bwd_prop_gaussianParticle_Parameters_x24init_0(&_S395, &_S396, &_S397, &_S398, &_S399);
    gaussianParticle_RawParameters_0 _S400 = gaussianParticle_RawParameters_x24_syn_dzero_0();
    (&_S400)->density_1 = _S398.differential_0;
    (&_S400)->inverseFactor_1 = _S397.differential_0;
    (&_S400)->factor_1 = _S396.differential_0;
    (&_S400)->position_1 = _S395.differential_0;
    gaussianParticle_RawParameters_0 _S401 = _S400;
    if(commonParameters_6.parametersBuffer_0.exclusiveGradient_0)
    {
        gaussianParticle_RawParameters_0 * _S402 = commonParameters_6.parametersBuffer_0._gradPtr_0 + particleIdx_6;
        _S402->density_1 = _S402->density_1 + _S401.density_1;
        _S402->position_1 = _S402->position_1 + _S401.position_1;
        _S402->quaternion_0 = _S402->quaternion_0 + _S401.quaternion_0;
        _S402->scale_0 = _S402->scale_0 + _S401.scale_0;
        _S402->factor_1 = _S402->factor_1 + _S401.factor_1;
        _S402->inverseFactor_1 = _S402->inverseFactor_1 + _S401.inverseFactor_1;
    }
    else
    {
        gaussianParticle_RawParameters_0 * _S403 = commonParameters_6.parametersBuffer_0._gradPtr_0 + particleIdx_6;
        float _S404 = atomicAdd(&_S403->density_1, _S401.density_1);
        float _S405 = atomicAdd(&((&_S403->position_1)->x), _S401.position_1.x);
        float _S406 = atomicAdd(&((&_S403->position_1)->y), _S401.position_1.y);
        float _S407 = atomicAdd(&((&_S403->position_1)->z), _S401.position_1.z);
        float _S408 = atomicAdd(&((&_S403->quaternion_0)->x), _S401.quaternion_0.x);
        float _S409 = atomicAdd(&((&_S403->quaternion_0)->y), _S401.quaternion_0.y);
        float _S410 = atomicAdd(&((&_S403->quaternion_0)->z), _S401.quaternion_0.z);
        float _S411 = atomicAdd(&((&_S403->quaternion_0)->w), _S401.quaternion_0.w);
        float _S412 = atomicAdd(&((&_S403->scale_0)->x), _S401.scale_0.x);
        float _S413 = atomicAdd(&((&_S403->scale_0)->y), _S401.scale_0.y);
        float _S414 = atomicAdd(&((&_S403->scale_0)->z), _S401.scale_0.z);
        float _S415 = atomicAdd(&(((&_S403->factor_1)->rows + (int(0)))->x), _S401.factor_1.rows[int(0)].x);
        float _S416 = atomicAdd(&(((&_S403->inverseFactor_1)->rows + (int(0)))->x), _S401.inverseFactor_1.rows[int(0)].x);
        float _S417 = atomicAdd(&(((&_S403->factor_1)->rows + (int(0)))->y), _S401.factor_1.rows[int(0)].y);
        float _S418 = atomicAdd(&(((&_S403->inverseFactor_1)->rows + (int(0)))->y), _S401.inverseFactor_1.rows[int(0)].y);
        float _S419 = atomicAdd(&(((&_S403->factor_1)->rows + (int(0)))->z), _S401.factor_1.rows[int(0)].z);
        float _S420 = atomicAdd(&(((&_S403->inverseFactor_1)->rows + (int(0)))->z), _S401.inverseFactor_1.rows[int(0)].z);
        float _S421 = atomicAdd(&(((&_S403->factor_1)->rows + (int(1)))->x), _S401.factor_1.rows[int(1)].x);
        float _S422 = atomicAdd(&(((&_S403->inverseFactor_1)->rows + (int(1)))->x), _S401.inverseFactor_1.rows[int(1)].x);
        float _S423 = atomicAdd(&(((&_S403->factor_1)->rows + (int(1)))->y), _S401.factor_1.rows[int(1)].y);
        float _S424 = atomicAdd(&(((&_S403->inverseFactor_1)->rows + (int(1)))->y), _S401.inverseFactor_1.rows[int(1)].y);
        float _S425 = atomicAdd(&(((&_S403->factor_1)->rows + (int(1)))->z), _S401.factor_1.rows[int(1)].z);
        float _S426 = atomicAdd(&(((&_S403->inverseFactor_1)->rows + (int(1)))->z), _S401.inverseFactor_1.rows[int(1)].z);
        float _S427 = atomicAdd(&(((&_S403->factor_1)->rows + (int(2)))->x), _S401.factor_1.rows[int(2)].x);
        float _S428 = atomicAdd(&(((&_S403->inverseFactor_1)->rows + (int(2)))->x), _S401.inverseFactor_1.rows[int(2)].x);
        float _S429 = atomicAdd(&(((&_S403->factor_1)->rows + (int(2)))->y), _S401.factor_1.rows[int(2)].y);
        float _S430 = atomicAdd(&(((&_S403->inverseFactor_1)->rows + (int(2)))->y), _S401.inverseFactor_1.rows[int(2)].y);
        float _S431 = atomicAdd(&(((&_S403->factor_1)->rows + (int(2)))->z), _S401.factor_1.rows[int(2)].z);
        float _S432 = atomicAdd(&(((&_S403->inverseFactor_1)->rows + (int(2)))->z), _S401.inverseFactor_1.rows[int(2)].z);
    }
    return;
}

__device__ void _d_max_vector_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_10, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_5, float3  dOut_10)
{
    DiffPair_float_0 left_dp_3;
    (&left_dp_3)->primal_0 = (*dpx_10).primal_0.x;
    (&left_dp_3)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_3;
    (&right_dp_3)->primal_0 = (*dpy_5).primal_0.x;
    (&right_dp_3)->differential_0 = 0.0f;
    _d_max_0(&left_dp_3, &right_dp_3, dOut_10.x);
    float3  left_d_result_2;
    *&((&left_d_result_2)->x) = left_dp_3.differential_0;
    float3  right_d_result_2;
    *&((&right_d_result_2)->x) = right_dp_3.differential_0;
    DiffPair_float_0 left_dp_4;
    (&left_dp_4)->primal_0 = (*dpx_10).primal_0.y;
    (&left_dp_4)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_4;
    (&right_dp_4)->primal_0 = (*dpy_5).primal_0.y;
    (&right_dp_4)->differential_0 = 0.0f;
    _d_max_0(&left_dp_4, &right_dp_4, dOut_10.y);
    *&((&left_d_result_2)->y) = left_dp_4.differential_0;
    *&((&right_d_result_2)->y) = right_dp_4.differential_0;
    DiffPair_float_0 left_dp_5;
    (&left_dp_5)->primal_0 = (*dpx_10).primal_0.z;
    (&left_dp_5)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_5;
    (&right_dp_5)->primal_0 = (*dpy_5).primal_0.z;
    (&right_dp_5)->differential_0 = 0.0f;
    _d_max_0(&left_dp_5, &right_dp_5, dOut_10.z);
    *&((&left_d_result_2)->z) = left_dp_5.differential_0;
    *&((&right_d_result_2)->z) = right_dp_5.differential_0;
    dpx_10->primal_0 = (*dpx_10).primal_0;
    dpx_10->differential_0 = left_d_result_2;
    dpy_5->primal_0 = (*dpy_5).primal_0;
    dpy_5->differential_0 = right_d_result_2;
    return;
}

__device__ float3  max_0(float3  x_5, float3  y_1)
{
    float3  result_8;
    int i_2 = int(0);
    for(;;)
    {
        if(i_2 < int(3))
        {
        }
        else
        {
            break;
        }
        *_slang_vector_get_element_ptr(&result_8, i_2) = (F32_max((_slang_vector_get_element(x_5, i_2)), (_slang_vector_get_element(y_1, i_2))));
        i_2 = i_2 + int(1);
    }
    return result_8;
}

__device__ float3  sphericalHarmonics_decode_0(int degree_0, FixedArray<float3 , 16>  * coefficients_0, float3  direction_0)
{
    float3  features_0 = make_float3 (0.282094806432724f) * (*coefficients_0)[int(0)];
    float3  features_1;
    if(degree_0 > int(0))
    {
        float x_6 = direction_0.x;
        float y_2 = direction_0.y;
        float z_0 = direction_0.z;
        float3  features_2 = features_0 - make_float3 (0.48860251903533936f * y_2) * (*coefficients_0)[int(1)] + make_float3 (0.48860251903533936f * z_0) * (*coefficients_0)[int(2)] - make_float3 (0.48860251903533936f * x_6) * (*coefficients_0)[int(3)];
        if(degree_0 > int(1))
        {
            float xx_0 = x_6 * x_6;
            float yy_0 = y_2 * y_2;
            float zz_0 = z_0 * z_0;
            float xy_0 = x_6 * y_2;
            float _S433 = 2.0f * zz_0;
            float _S434 = xx_0 - yy_0;
            float3  features_3 = features_2 + make_float3 (1.09254848957061768f * xy_0) * (*coefficients_0)[int(4)] + make_float3 (-1.09254848957061768f * (y_2 * z_0)) * (*coefficients_0)[int(5)] + make_float3 (0.31539157032966614f * (_S433 - xx_0 - yy_0)) * (*coefficients_0)[int(6)] + make_float3 (-1.09254848957061768f * (x_6 * z_0)) * (*coefficients_0)[int(7)] + make_float3 (0.54627424478530884f * _S434) * (*coefficients_0)[int(8)];
            if(degree_0 > int(2))
            {
                float _S435 = 3.0f * xx_0;
                float _S436 = 4.0f * zz_0 - xx_0 - yy_0;
                float _S437 = 3.0f * yy_0;
                features_1 = features_3 + make_float3 (-0.59004360437393188f * y_2 * (_S435 - yy_0)) * (*coefficients_0)[int(9)] + make_float3 (2.89061141014099121f * xy_0 * z_0) * (*coefficients_0)[int(10)] + make_float3 (-0.4570457935333252f * y_2 * _S436) * (*coefficients_0)[int(11)] + make_float3 (0.37317633628845215f * z_0 * (_S433 - _S435 - _S437)) * (*coefficients_0)[int(12)] + make_float3 (-0.4570457935333252f * x_6 * _S436) * (*coefficients_0)[int(13)] + make_float3 (1.44530570507049561f * z_0 * _S434) * (*coefficients_0)[int(14)] + make_float3 (-0.59004360437393188f * x_6 * (xx_0 - _S437)) * (*coefficients_0)[int(15)];
            }
            else
            {
                features_1 = features_3;
            }
        }
        else
        {
            features_1 = features_2;
        }
    }
    else
    {
        features_1 = features_0;
    }
    int3  _S438 = make_int3 (int(0));
    float3  _S439 = make_float3 ((float)_S438.x, (float)_S438.y, (float)_S438.z);
    return max_0(features_1 + make_float3 (0.5f), _S439);
}

struct shRadiativeParticle_ParametersBuffer_0
{
    float3  * _dataPtr_1;
    float3  * _gradPtr_1;
    bool exclusiveGradient_1;
};

__device__ void particleFeaturesFromBuffer(uint particleIdx_7, float * radianceBufferPtr_0, int auxParam_0, float3  incidentDirection_0, float3  canonicalPosition_0, FixedArray<float, 3>  * features_4)
{
    shRadiativeParticle_ParametersBuffer_0 parametersBuffer_1;
    (&parametersBuffer_1)->_dataPtr_1 = (slang_bit_cast<float3  *>((slang_bit_cast<ulonglong>(radianceBufferPtr_0))));
    (&parametersBuffer_1)->_gradPtr_1 = nullptr;
    (&parametersBuffer_1)->exclusiveGradient_1 = false;
    float3  _S440 = make_float3 (0.0f);
    FixedArray<float3 , 16>  _S441 = {
        _S440, _S440, _S440, _S440, _S440, _S440, _S440, _S440, _S440, _S440, _S440, _S440, _S440, _S440, _S440, _S440
    };
    shRadiativeParticle_Parameters_0 parameters_3;
    (&parameters_3)->sphCoefficients_0 = _S441;
    uint _S442 = particleIdx_7 * 16U;
    (&parameters_3)->sphCoefficients_0[int(0)] = *(parametersBuffer_1._dataPtr_1 + _S442);
    (&parameters_3)->sphCoefficients_0[int(1)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 1U));
    (&parameters_3)->sphCoefficients_0[int(2)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 2U));
    (&parameters_3)->sphCoefficients_0[int(3)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 3U));
    (&parameters_3)->sphCoefficients_0[int(4)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 4U));
    (&parameters_3)->sphCoefficients_0[int(5)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 5U));
    (&parameters_3)->sphCoefficients_0[int(6)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 6U));
    (&parameters_3)->sphCoefficients_0[int(7)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 7U));
    (&parameters_3)->sphCoefficients_0[int(8)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 8U));
    (&parameters_3)->sphCoefficients_0[int(9)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 9U));
    (&parameters_3)->sphCoefficients_0[int(10)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 10U));
    (&parameters_3)->sphCoefficients_0[int(11)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 11U));
    (&parameters_3)->sphCoefficients_0[int(12)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 12U));
    (&parameters_3)->sphCoefficients_0[int(13)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 13U));
    (&parameters_3)->sphCoefficients_0[int(14)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 14U));
    (&parameters_3)->sphCoefficients_0[int(15)] = *(parametersBuffer_1._dataPtr_1 + (_S442 + 15U));
    FixedArray<float3 , 16>  _S443 = (&parameters_3)->sphCoefficients_0;
    float3  _S444 = sphericalHarmonics_decode_0(auxParam_0, &_S443, incidentDirection_0);
    (*features_4)[int(0)] = _S444.x;
    (*features_4)[int(1)] = _S444.y;
    (*features_4)[int(2)] = _S444.z;
    return;
}

__device__ void particleFeaturesIntegrateFwd(float weight_2, FixedArray<float, 3>  features_5, FixedArray<float, 3>  * integratedFeatures_0)
{
    float3  featuresVec_0;
    *&((&featuresVec_0)->x) = features_5[int(0)];
    *&((&featuresVec_0)->y) = features_5[int(1)];
    *&((&featuresVec_0)->z) = features_5[int(2)];
    float3  integratedVec_0;
    *&((&integratedVec_0)->x) = (*integratedFeatures_0)[int(0)];
    *&((&integratedVec_0)->y) = (*integratedFeatures_0)[int(1)];
    *&((&integratedVec_0)->z) = (*integratedFeatures_0)[int(2)];
    float3  _S445 = featuresVec_0;
    if(weight_2 > 0.0f)
    {
        integratedVec_0 = integratedVec_0 + _S445 * make_float3 (weight_2);
    }
    (*integratedFeatures_0)[int(0)] = *&((&integratedVec_0)->x);
    (*integratedFeatures_0)[int(1)] = *&((&integratedVec_0)->y);
    (*integratedFeatures_0)[int(2)] = *&((&integratedVec_0)->z);
    return;
}

__device__ void particleFeaturesIntegrateFwdFromBuffer(float3  incidentDirection_1, float3  canonicalPosition_1, float weight_3, uint particleIdx_8, float * radianceBufferPtr_1, int auxParam_1, FixedArray<float, 3>  * integratedFeatures_1)
{
    shRadiativeParticle_ParametersBuffer_0 parametersBuffer_2;
    (&parametersBuffer_2)->_dataPtr_1 = (slang_bit_cast<float3  *>((slang_bit_cast<ulonglong>(radianceBufferPtr_1))));
    (&parametersBuffer_2)->_gradPtr_1 = nullptr;
    (&parametersBuffer_2)->exclusiveGradient_1 = false;
    float3  integratedVec_1;
    *&((&integratedVec_1)->x) = (*integratedFeatures_1)[int(0)];
    *&((&integratedVec_1)->y) = (*integratedFeatures_1)[int(1)];
    *&((&integratedVec_1)->z) = (*integratedFeatures_1)[int(2)];
    uint _S446 = uint(auxParam_1);
    float3  _S447 = make_float3 (0.0f);
    FixedArray<float3 , 16>  _S448 = {
        _S447, _S447, _S447, _S447, _S447, _S447, _S447, _S447, _S447, _S447, _S447, _S447, _S447, _S447, _S447, _S447
    };
    shRadiativeParticle_Parameters_0 parameters_4;
    (&parameters_4)->sphCoefficients_0 = _S448;
    uint _S449 = particleIdx_8 * 16U;
    (&parameters_4)->sphCoefficients_0[int(0)] = *(parametersBuffer_2._dataPtr_1 + _S449);
    (&parameters_4)->sphCoefficients_0[int(1)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 1U));
    (&parameters_4)->sphCoefficients_0[int(2)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 2U));
    (&parameters_4)->sphCoefficients_0[int(3)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 3U));
    (&parameters_4)->sphCoefficients_0[int(4)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 4U));
    (&parameters_4)->sphCoefficients_0[int(5)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 5U));
    (&parameters_4)->sphCoefficients_0[int(6)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 6U));
    (&parameters_4)->sphCoefficients_0[int(7)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 7U));
    (&parameters_4)->sphCoefficients_0[int(8)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 8U));
    (&parameters_4)->sphCoefficients_0[int(9)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 9U));
    (&parameters_4)->sphCoefficients_0[int(10)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 10U));
    (&parameters_4)->sphCoefficients_0[int(11)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 11U));
    (&parameters_4)->sphCoefficients_0[int(12)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 12U));
    (&parameters_4)->sphCoefficients_0[int(13)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 13U));
    (&parameters_4)->sphCoefficients_0[int(14)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 14U));
    (&parameters_4)->sphCoefficients_0[int(15)] = *(parametersBuffer_2._dataPtr_1 + (_S449 + 15U));
    bool _S450 = weight_3 > 0.0f;
    if(_S450)
    {
        int _S451 = int(_S446);
        FixedArray<float3 , 16>  _S452 = (&parameters_4)->sphCoefficients_0;
        float3  _S453 = sphericalHarmonics_decode_0(_S451, &_S452, incidentDirection_1);
        if(_S450)
        {
            integratedVec_1 = integratedVec_1 + _S453 * make_float3 (weight_3);
        }
    }
    (*integratedFeatures_1)[int(0)] = *&((&integratedVec_1)->x);
    (*integratedFeatures_1)[int(1)] = *&((&integratedVec_1)->y);
    (*integratedFeatures_1)[int(2)] = *&((&integratedVec_1)->z);
    return;
}

struct s_bwd_prop_shRadiativeParticle_integrateRadiance_Intermediates_0
{
    float3  _S454;
};

__device__ void particleFeaturesIntegrateBwd(float alpha_7, float * alphaGrad_2, FixedArray<float, 3>  features_6, FixedArray<float, 3>  * featuresGrad_0, FixedArray<float, 3>  * integratedFeatures_2, FixedArray<float, 3>  * integratedFeaturesGrad_0)
{
    bool _S455 = alpha_7 > 0.0f;
    if(_S455)
    {
        float3  featuresVec_1;
        *&((&featuresVec_1)->x) = features_6[int(0)];
        float3  featuresGradVec_0;
        *&((&featuresGradVec_0)->x) = (*featuresGrad_0)[int(0)];
        *&((&featuresVec_1)->y) = features_6[int(1)];
        *&((&featuresGradVec_0)->y) = (*featuresGrad_0)[int(1)];
        *&((&featuresVec_1)->z) = features_6[int(2)];
        *&((&featuresGradVec_0)->z) = (*featuresGrad_0)[int(2)];
        float3  integratedVec_2;
        *&((&integratedVec_2)->x) = (*integratedFeatures_2)[int(0)];
        float3  integratedGradVec_0;
        *&((&integratedGradVec_0)->x) = (*integratedFeaturesGrad_0)[int(0)];
        *&((&integratedVec_2)->y) = (*integratedFeatures_2)[int(1)];
        *&((&integratedGradVec_0)->y) = (*integratedFeaturesGrad_0)[int(1)];
        *&((&integratedVec_2)->z) = (*integratedFeatures_2)[int(2)];
        *&((&integratedGradVec_0)->z) = (*integratedFeaturesGrad_0)[int(2)];
        float3  _S456 = featuresVec_1;
        float3  _S457 = (integratedVec_2 - featuresVec_1 * make_float3 (alpha_7)) * make_float3 (1.0f / (1.0f - alpha_7));
        integratedVec_2 = _S457;
        float3  _S458 = integratedGradVec_0;
        float3  _S459 = make_float3 (0.0f);
        s_bwd_prop_shRadiativeParticle_integrateRadiance_Intermediates_0 _S460;
        (&_S460)->_S454 = _S459;
        (&_S460)->_S454 = _S457;
        s_bwd_prop_shRadiativeParticle_integrateRadiance_Intermediates_0 _S461 = _S460;
        float3  _S462;
        if(_S455)
        {
            _S462 = make_float3 (alpha_7);
        }
        else
        {
            _S462 = _S459;
        }
        float3  dpintegratedRadiance_0;
        float _S463;
        if(_S455)
        {
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S464;
            (&_S464)->primal_0 = _S461._S454;
            (&_S464)->differential_0 = _S459;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S465;
            (&_S465)->primal_0 = _S456;
            (&_S465)->differential_0 = _S459;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S466;
            (&_S466)->primal_0 = _S462;
            (&_S466)->differential_0 = _S459;
            s_bwd_prop_lerp_0(&_S464, &_S465, &_S466, _S458);
            _S463 = _S466.differential_0.x + _S466.differential_0.y + _S466.differential_0.z;
            _S462 = _S465.differential_0;
            dpintegratedRadiance_0 = _S464.differential_0;
        }
        else
        {
            _S463 = 0.0f;
            _S462 = _S459;
            dpintegratedRadiance_0 = _S458;
        }
        *alphaGrad_2 = _S463;
        (*featuresGrad_0)[int(0)] = _S462.x;
        (*featuresGrad_0)[int(1)] = _S462.y;
        (*featuresGrad_0)[int(2)] = _S462.z;
        (*integratedFeaturesGrad_0)[int(0)] = dpintegratedRadiance_0.x;
        (*integratedFeaturesGrad_0)[int(1)] = dpintegratedRadiance_0.y;
        (*integratedFeaturesGrad_0)[int(2)] = dpintegratedRadiance_0.z;
        (*integratedFeatures_2)[int(0)] = *&((&integratedVec_2)->x);
        (*integratedFeatures_2)[int(1)] = *&((&integratedVec_2)->y);
        (*integratedFeatures_2)[int(2)] = *&((&integratedVec_2)->z);
    }
    return;
}

struct s_bwd_prop_shRadiativeParticle_integrateRadianceFromBuffer_Intermediates_0
{
    float3  _S467;
    shRadiativeParticle_Parameters_0 _S468;
};

__device__ float3  s_primal_ctx_max_0(float3  _S469, float3  _S470)
{
    return max_0(_S469, _S470);
}

__device__ float3  s_primal_ctx_sphericalHarmonics_decode_0(int degree_1, FixedArray<float3 , 16>  * dpcoefficients_0, float3  dpdirection_0)
{
    float3  features_7 = make_float3 (0.282094806432724f) * (*dpcoefficients_0)[int(0)];
    float3  features_8;
    if(degree_1 > int(0))
    {
        float x_7 = dpdirection_0.x;
        float y_3 = dpdirection_0.y;
        float z_1 = dpdirection_0.z;
        float3  features_9 = features_7 - make_float3 (0.48860251903533936f * y_3) * (*dpcoefficients_0)[int(1)] + make_float3 (0.48860251903533936f * z_1) * (*dpcoefficients_0)[int(2)] - make_float3 (0.48860251903533936f * x_7) * (*dpcoefficients_0)[int(3)];
        if(degree_1 > int(1))
        {
            float xx_1 = x_7 * x_7;
            float yy_1 = y_3 * y_3;
            float zz_1 = z_1 * z_1;
            float xy_1 = x_7 * y_3;
            float _S471 = 2.0f * zz_1;
            float _S472 = xx_1 - yy_1;
            float3  features_10 = features_9 + make_float3 (1.09254848957061768f * xy_1) * (*dpcoefficients_0)[int(4)] + make_float3 (-1.09254848957061768f * (y_3 * z_1)) * (*dpcoefficients_0)[int(5)] + make_float3 (0.31539157032966614f * (_S471 - xx_1 - yy_1)) * (*dpcoefficients_0)[int(6)] + make_float3 (-1.09254848957061768f * (x_7 * z_1)) * (*dpcoefficients_0)[int(7)] + make_float3 (0.54627424478530884f * _S472) * (*dpcoefficients_0)[int(8)];
            if(degree_1 > int(2))
            {
                float _S473 = 3.0f * xx_1;
                float _S474 = 4.0f * zz_1 - xx_1 - yy_1;
                float _S475 = 3.0f * yy_1;
                features_8 = features_10 + make_float3 (-0.59004360437393188f * y_3 * (_S473 - yy_1)) * (*dpcoefficients_0)[int(9)] + make_float3 (2.89061141014099121f * xy_1 * z_1) * (*dpcoefficients_0)[int(10)] + make_float3 (-0.4570457935333252f * y_3 * _S474) * (*dpcoefficients_0)[int(11)] + make_float3 (0.37317633628845215f * z_1 * (_S471 - _S473 - _S475)) * (*dpcoefficients_0)[int(12)] + make_float3 (-0.4570457935333252f * x_7 * _S474) * (*dpcoefficients_0)[int(13)] + make_float3 (1.44530570507049561f * z_1 * _S472) * (*dpcoefficients_0)[int(14)] + make_float3 (-0.59004360437393188f * x_7 * (xx_1 - _S475)) * (*dpcoefficients_0)[int(15)];
            }
            else
            {
                features_8 = features_10;
            }
        }
        else
        {
            features_8 = features_9;
        }
    }
    else
    {
        features_8 = features_7;
    }
    int3  _S476 = make_int3 (int(0));
    float3  _S477 = make_float3 ((float)_S476.x, (float)_S476.y, (float)_S476.z);
    return s_primal_ctx_max_0(features_8 + make_float3 (0.5f), _S477);
}

struct DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0
{
    FixedArray<float3 , 16>  primal_0;
    FixedArray<float3 , 16>  differential_0;
};

__device__ void s_bwd_prop_max_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S478, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S479, float3  _S480)
{
    _d_max_vector_0(_S478, _S479, _S480);
    return;
}

__device__ void s_bwd_prop_sphericalHarmonics_decode_0(int degree_2, DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0 * dpcoefficients_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpdirection_1, float3  _s_dOut_3)
{
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S481 = *dpdirection_1;
    float3  _S482 = make_float3 (0.0f);
    float3  features_11 = make_float3 (0.282094806432724f) * dpcoefficients_1->primal_0[int(0)];
    bool _S483 = degree_2 > int(0);
    float3  features_12;
    float3  _S484;
    float3  _S485;
    float3  _S486;
    float3  _S487;
    float3  _S488;
    float3  _S489;
    float3  _S490;
    float3  _S491;
    float3  _S492;
    float3  _S493;
    float3  _S494;
    float3  _S495;
    float3  _S496;
    float3  _S497;
    float3  _S498;
    float3  _S499;
    float3  _S500;
    float3  _S501;
    float3  _S502;
    float3  _S503;
    float3  _S504;
    float3  _S505;
    float3  _S506;
    float3  _S507;
    float3  _S508;
    float3  _S509;
    float3  _S510;
    float3  _S511;
    float3  _S512;
    float3  _S513;
    float _S514;
    float _S515;
    float _S516;
    float _S517;
    float _S518;
    float _S519;
    float _S520;
    float _S521;
    float _S522;
    float _S523;
    float _S524;
    float _S525;
    float _S526;
    float _S527;
    float _S528;
    bool _S529;
    bool _S530;
    if(_S483)
    {
        float x_8 = _S481.primal_0.x;
        float y_4 = _S481.primal_0.y;
        float z_2 = _S481.primal_0.z;
        float _S531 = 0.48860251903533936f * y_4;
        float3  _S532 = make_float3 (_S531);
        float3  _S533 = dpcoefficients_1->primal_0[int(1)];
        float _S534 = 0.48860251903533936f * z_2;
        float3  _S535 = make_float3 (_S534);
        float3  _S536 = dpcoefficients_1->primal_0[int(2)];
        float _S537 = 0.48860251903533936f * x_8;
        float3  _S538 = make_float3 (_S537);
        float3  _S539 = dpcoefficients_1->primal_0[int(3)];
        float3  features_13 = features_11 - make_float3 (_S531) * dpcoefficients_1->primal_0[int(1)] + make_float3 (_S534) * dpcoefficients_1->primal_0[int(2)] - make_float3 (_S537) * dpcoefficients_1->primal_0[int(3)];
        bool _S540 = degree_2 > int(1);
        if(_S540)
        {
            float xx_2 = x_8 * x_8;
            float yy_2 = y_4 * y_4;
            float zz_2 = z_2 * z_2;
            float xy_2 = x_8 * y_4;
            float _S541 = 1.09254848957061768f * xy_2;
            float3  _S542 = make_float3 (_S541);
            float3  _S543 = dpcoefficients_1->primal_0[int(4)];
            float _S544 = -1.09254848957061768f * (y_4 * z_2);
            float3  _S545 = make_float3 (_S544);
            float3  _S546 = dpcoefficients_1->primal_0[int(5)];
            float _S547 = 2.0f * zz_2;
            float _S548 = 0.31539157032966614f * (_S547 - xx_2 - yy_2);
            float3  _S549 = make_float3 (_S548);
            float3  _S550 = dpcoefficients_1->primal_0[int(6)];
            float _S551 = -1.09254848957061768f * (x_8 * z_2);
            float3  _S552 = make_float3 (_S551);
            float3  _S553 = dpcoefficients_1->primal_0[int(7)];
            float _S554 = xx_2 - yy_2;
            float _S555 = 0.54627424478530884f * _S554;
            float3  _S556 = make_float3 (_S555);
            float3  _S557 = dpcoefficients_1->primal_0[int(8)];
            float3  features_14 = features_13 + make_float3 (_S541) * dpcoefficients_1->primal_0[int(4)] + make_float3 (_S544) * dpcoefficients_1->primal_0[int(5)] + make_float3 (_S548) * dpcoefficients_1->primal_0[int(6)] + make_float3 (_S551) * dpcoefficients_1->primal_0[int(7)] + make_float3 (_S555) * dpcoefficients_1->primal_0[int(8)];
            bool _S558 = degree_2 > int(2);
            if(_S558)
            {
                float _S559 = -0.59004360437393188f * y_4;
                float _S560 = 3.0f * xx_2;
                float _S561 = _S560 - yy_2;
                float _S562 = _S559 * _S561;
                float3  _S563 = make_float3 (_S562);
                float _S564 = 2.89061141014099121f * xy_2;
                float _S565 = _S564 * z_2;
                float3  _S566 = make_float3 (_S565);
                float _S567 = -0.4570457935333252f * y_4;
                float _S568 = 4.0f * zz_2 - xx_2 - yy_2;
                float _S569 = _S567 * _S568;
                float3  _S570 = make_float3 (_S569);
                float _S571 = 0.37317633628845215f * z_2;
                float _S572 = 3.0f * yy_2;
                float _S573 = _S547 - _S560 - _S572;
                float _S574 = _S571 * _S573;
                float3  _S575 = make_float3 (_S574);
                float _S576 = -0.4570457935333252f * x_8;
                float _S577 = _S576 * _S568;
                float3  _S578 = make_float3 (_S577);
                float _S579 = 1.44530570507049561f * z_2;
                float _S580 = _S579 * _S554;
                float3  _S581 = make_float3 (_S580);
                float _S582 = -0.59004360437393188f * x_8;
                float _S583 = xx_2 - _S572;
                float _S584 = _S582 * _S583;
                float3  _S585 = make_float3 (_S584);
                features_12 = features_14 + make_float3 (_S562) * dpcoefficients_1->primal_0[int(9)] + make_float3 (_S565) * dpcoefficients_1->primal_0[int(10)] + make_float3 (_S569) * dpcoefficients_1->primal_0[int(11)] + make_float3 (_S574) * dpcoefficients_1->primal_0[int(12)] + make_float3 (_S577) * dpcoefficients_1->primal_0[int(13)] + make_float3 (_S580) * dpcoefficients_1->primal_0[int(14)] + make_float3 (_S584) * dpcoefficients_1->primal_0[int(15)];
                _S484 = _S585;
                _S485 = dpcoefficients_1->primal_0[int(15)];
                _S514 = _S582;
                _S515 = _S583;
                _S486 = _S581;
                _S487 = dpcoefficients_1->primal_0[int(14)];
                _S516 = _S579;
                _S488 = _S578;
                _S489 = dpcoefficients_1->primal_0[int(13)];
                _S517 = _S576;
                _S518 = _S568;
                _S490 = _S575;
                _S491 = dpcoefficients_1->primal_0[int(12)];
                _S519 = _S571;
                _S520 = _S573;
                _S492 = _S570;
                _S493 = dpcoefficients_1->primal_0[int(11)];
                _S521 = _S567;
                _S494 = _S566;
                _S495 = dpcoefficients_1->primal_0[int(10)];
                _S522 = _S564;
                _S496 = _S563;
                _S497 = dpcoefficients_1->primal_0[int(9)];
                _S523 = _S559;
                _S524 = _S561;
            }
            else
            {
                features_12 = features_14;
                _S484 = _S482;
                _S485 = _S482;
                _S514 = 0.0f;
                _S515 = 0.0f;
                _S486 = _S482;
                _S487 = _S482;
                _S516 = 0.0f;
                _S488 = _S482;
                _S489 = _S482;
                _S517 = 0.0f;
                _S518 = 0.0f;
                _S490 = _S482;
                _S491 = _S482;
                _S519 = 0.0f;
                _S520 = 0.0f;
                _S492 = _S482;
                _S493 = _S482;
                _S521 = 0.0f;
                _S494 = _S482;
                _S495 = _S482;
                _S522 = 0.0f;
                _S496 = _S482;
                _S497 = _S482;
                _S523 = 0.0f;
                _S524 = 0.0f;
            }
            float _S586 = _S517;
            float _S587 = _S518;
            float _S588 = _S519;
            float _S589 = _S520;
            float _S590 = _S521;
            float _S591 = _S522;
            float _S592 = _S523;
            float _S593 = _S524;
            _S529 = _S558;
            _S517 = _S554;
            _S518 = _S586;
            _S519 = _S587;
            _S520 = _S588;
            _S521 = _S589;
            _S522 = _S590;
            _S523 = _S591;
            _S524 = _S592;
            _S525 = _S593;
            _S498 = _S556;
            _S499 = _S557;
            _S500 = _S552;
            _S501 = _S553;
            _S502 = _S549;
            _S503 = _S550;
            _S504 = _S545;
            _S505 = _S546;
            _S506 = _S542;
            _S507 = _S543;
        }
        else
        {
            features_12 = features_13;
            _S529 = false;
            _S484 = _S482;
            _S485 = _S482;
            _S514 = 0.0f;
            _S515 = 0.0f;
            _S486 = _S482;
            _S487 = _S482;
            _S516 = 0.0f;
            _S517 = 0.0f;
            _S488 = _S482;
            _S489 = _S482;
            _S518 = 0.0f;
            _S519 = 0.0f;
            _S490 = _S482;
            _S491 = _S482;
            _S520 = 0.0f;
            _S521 = 0.0f;
            _S492 = _S482;
            _S493 = _S482;
            _S522 = 0.0f;
            _S494 = _S482;
            _S495 = _S482;
            _S523 = 0.0f;
            _S496 = _S482;
            _S497 = _S482;
            _S524 = 0.0f;
            _S525 = 0.0f;
            _S498 = _S482;
            _S499 = _S482;
            _S500 = _S482;
            _S501 = _S482;
            _S502 = _S482;
            _S503 = _S482;
            _S504 = _S482;
            _S505 = _S482;
            _S506 = _S482;
            _S507 = _S482;
        }
        bool _S594 = _S529;
        float _S595 = _S524;
        float _S596 = _S525;
        _S529 = _S540;
        _S530 = _S594;
        _S524 = z_2;
        _S525 = _S595;
        _S526 = _S596;
        _S527 = x_8;
        _S528 = y_4;
        _S508 = _S538;
        _S509 = _S539;
        _S510 = _S535;
        _S511 = _S536;
        _S512 = _S532;
        _S513 = _S533;
    }
    else
    {
        features_12 = features_11;
        _S529 = false;
        _S530 = false;
        _S484 = _S482;
        _S485 = _S482;
        _S514 = 0.0f;
        _S515 = 0.0f;
        _S486 = _S482;
        _S487 = _S482;
        _S516 = 0.0f;
        _S517 = 0.0f;
        _S488 = _S482;
        _S489 = _S482;
        _S518 = 0.0f;
        _S519 = 0.0f;
        _S490 = _S482;
        _S491 = _S482;
        _S520 = 0.0f;
        _S521 = 0.0f;
        _S492 = _S482;
        _S493 = _S482;
        _S522 = 0.0f;
        _S494 = _S482;
        _S495 = _S482;
        _S523 = 0.0f;
        _S524 = 0.0f;
        _S496 = _S482;
        _S497 = _S482;
        _S525 = 0.0f;
        _S526 = 0.0f;
        _S498 = _S482;
        _S499 = _S482;
        _S500 = _S482;
        _S501 = _S482;
        _S502 = _S482;
        _S503 = _S482;
        _S504 = _S482;
        _S505 = _S482;
        _S506 = _S482;
        _S507 = _S482;
        _S527 = 0.0f;
        _S528 = 0.0f;
        _S508 = _S482;
        _S509 = _S482;
        _S510 = _S482;
        _S511 = _S482;
        _S512 = _S482;
        _S513 = _S482;
    }
    float3  _S597 = features_12 + make_float3 (0.5f);
    int3  _S598 = make_int3 (int(0));
    float3  _S599 = make_float3 ((float)_S598.x, (float)_S598.y, (float)_S598.z);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S600;
    (&_S600)->primal_0 = _S597;
    (&_S600)->differential_0 = _S482;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S601;
    (&_S601)->primal_0 = _S599;
    (&_S601)->differential_0 = _S482;
    s_bwd_prop_max_0(&_S600, &_S601, _s_dOut_3);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S602 = _S600;
    FixedArray<float3 , 16>  _S603;
    if(_S483)
    {
        if(_S529)
        {
            if(_S530)
            {
                float3  _S604 = _S484 * _S602.differential_0;
                float3  _S605 = _S485 * _S602.differential_0;
                float _S606 = _S605.x + _S605.y + _S605.z;
                float _S607 = _S514 * _S606;
                float3  _S608 = _S486 * _S602.differential_0;
                float3  _S609 = _S487 * _S602.differential_0;
                float _S610 = _S609.x + _S609.y + _S609.z;
                float _S611 = _S516 * _S610;
                float _S612 = 1.44530570507049561f * (_S517 * _S610);
                float3  _S613 = _S488 * _S602.differential_0;
                float3  _S614 = _S489 * _S602.differential_0;
                float _S615 = _S614.x + _S614.y + _S614.z;
                float3  _S616 = _S490 * _S602.differential_0;
                float3  _S617 = _S491 * _S602.differential_0;
                float _S618 = _S617.x + _S617.y + _S617.z;
                float _S619 = _S520 * _S618;
                float _S620 = - _S619;
                float _S621 = 3.0f * (- _S607 + _S620);
                float _S622 = 0.37317633628845215f * (_S521 * _S618);
                float3  _S623 = _S492 * _S602.differential_0;
                float3  _S624 = _S493 * _S602.differential_0;
                float _S625 = _S624.x + _S624.y + _S624.z;
                float _S626 = _S518 * _S615 + _S522 * _S625;
                float _S627 = - _S626;
                float _S628 = 4.0f * _S626;
                float _S629 = -0.4570457935333252f * (_S519 * _S625);
                float3  _S630 = _S494 * _S602.differential_0;
                float3  _S631 = _S495 * _S602.differential_0;
                float _S632 = _S631.x + _S631.y + _S631.z;
                float _S633 = _S523 * _S632;
                float _S634 = 2.89061141014099121f * (_S524 * _S632);
                float3  _S635 = _S496 * _S602.differential_0;
                float3  _S636 = _S497 * _S602.differential_0;
                float _S637 = _S636.x + _S636.y + _S636.z;
                float _S638 = _S525 * _S637;
                float _S639 = - _S638;
                float _S640 = 3.0f * (_S620 + _S638);
                float _S641 = -0.59004360437393188f * (_S526 * _S637);
                float _S642 = -0.59004360437393188f * (_S515 * _S606) + -0.4570457935333252f * (_S519 * _S615);
                FixedArray<float3 , 16>  _S643;
                _S643[int(0)] = _S482;
                _S643[int(1)] = _S482;
                _S643[int(2)] = _S482;
                _S643[int(3)] = _S482;
                _S643[int(4)] = _S482;
                _S643[int(5)] = _S482;
                _S643[int(6)] = _S482;
                _S643[int(7)] = _S482;
                _S643[int(8)] = _S482;
                _S643[int(9)] = _S482;
                _S643[int(10)] = _S482;
                _S643[int(11)] = _S482;
                _S643[int(12)] = _S482;
                _S643[int(13)] = _S482;
                _S643[int(14)] = _S482;
                _S643[int(15)] = _S482;
                _S643[int(15)] = _S604;
                _S643[int(14)] = _S608;
                _S643[int(13)] = _S613;
                _S643[int(12)] = _S616;
                _S643[int(11)] = _S623;
                _S643[int(10)] = _S630;
                _S643[int(9)] = _S635;
                float _S644 = _S621 + _S627 + _S639;
                float _S645 = _S607 + _S627 + _S640;
                float _S646 = _S612 + _S622 + _S633;
                float _S647 = _S629 + _S641;
                _S514 = _S611;
                _S515 = _S619;
                _S516 = _S634;
                _S517 = _S628;
                _S518 = _S644;
                _S519 = _S645;
                _S603[int(0)] = _S643[int(0)];
                _S603[int(1)] = _S643[int(1)];
                _S603[int(2)] = _S643[int(2)];
                _S603[int(3)] = _S643[int(3)];
                _S603[int(4)] = _S643[int(4)];
                _S603[int(5)] = _S643[int(5)];
                _S603[int(6)] = _S643[int(6)];
                _S603[int(7)] = _S643[int(7)];
                _S603[int(8)] = _S643[int(8)];
                _S603[int(9)] = _S635;
                _S603[int(10)] = _S630;
                _S603[int(11)] = _S623;
                _S603[int(12)] = _S616;
                _S603[int(13)] = _S613;
                _S603[int(14)] = _S608;
                _S603[int(15)] = _S604;
                _S520 = _S642;
                _S521 = _S647;
                _S522 = _S646;
            }
            else
            {
                _S514 = 0.0f;
                _S515 = 0.0f;
                _S516 = 0.0f;
                _S517 = 0.0f;
                _S518 = 0.0f;
                _S519 = 0.0f;
                _S603[int(0)] = _S482;
                _S603[int(1)] = _S482;
                _S603[int(2)] = _S482;
                _S603[int(3)] = _S482;
                _S603[int(4)] = _S482;
                _S603[int(5)] = _S482;
                _S603[int(6)] = _S482;
                _S603[int(7)] = _S482;
                _S603[int(8)] = _S482;
                _S603[int(9)] = _S482;
                _S603[int(10)] = _S482;
                _S603[int(11)] = _S482;
                _S603[int(12)] = _S482;
                _S603[int(13)] = _S482;
                _S603[int(14)] = _S482;
                _S603[int(15)] = _S482;
                _S520 = 0.0f;
                _S521 = 0.0f;
                _S522 = 0.0f;
            }
            float3  _S648 = _S498 * _S602.differential_0;
            float3  _S649 = _S499 * _S602.differential_0;
            float _S650 = 0.54627424478530884f * (_S649.x + _S649.y + _S649.z) + _S514;
            float3  _S651 = _S500 * _S602.differential_0;
            float3  _S652 = _S501 * _S602.differential_0;
            float s_diff_xz_T_0 = -1.09254848957061768f * (_S652.x + _S652.y + _S652.z);
            float3  _S653 = _S502 * _S602.differential_0;
            float3  _S654 = _S503 * _S602.differential_0;
            float _S655 = 0.31539157032966614f * (_S654.x + _S654.y + _S654.z);
            float _S656 = - _S655;
            float3  _S657 = _S504 * _S602.differential_0;
            float3  _S658 = _S505 * _S602.differential_0;
            float s_diff_yz_T_0 = -1.09254848957061768f * (_S658.x + _S658.y + _S658.z);
            float3  _S659 = _S506 * _S602.differential_0;
            float3  _S660 = _S507 * _S602.differential_0;
            float _S661 = _S527 * s_diff_xz_T_0;
            float _S662 = _S524 * s_diff_xz_T_0;
            float _S663 = _S528 * s_diff_yz_T_0;
            float _S664 = _S524 * s_diff_yz_T_0;
            float _S665 = 1.09254848957061768f * (_S660.x + _S660.y + _S660.z) + _S516;
            float _S666 = _S527 * _S665;
            float _S667 = _S528 * _S665;
            float _S668 = _S524 * (2.0f * (_S655 + _S515) + _S517);
            float _S669 = _S528 * (- _S650 + _S656 + _S518);
            float _S670 = _S527 * (_S650 + _S656 + _S519);
            FixedArray<float3 , 16>  _S671;
            _S671[int(0)] = _S482;
            _S671[int(1)] = _S482;
            _S671[int(2)] = _S482;
            _S671[int(3)] = _S482;
            _S671[int(4)] = _S482;
            _S671[int(5)] = _S482;
            _S671[int(6)] = _S482;
            _S671[int(7)] = _S482;
            _S671[int(8)] = _S482;
            _S671[int(9)] = _S482;
            _S671[int(10)] = _S482;
            _S671[int(11)] = _S482;
            _S671[int(12)] = _S482;
            _S671[int(13)] = _S482;
            _S671[int(14)] = _S482;
            _S671[int(15)] = _S482;
            _S671[int(8)] = _S648;
            _S671[int(7)] = _S651;
            _S671[int(6)] = _S653;
            _S671[int(5)] = _S657;
            _S671[int(4)] = _S659;
            float3  _S672 = _S603[int(0)] + _S671[int(0)];
            float3  _S673 = _S603[int(1)] + _S671[int(1)];
            float3  _S674 = _S603[int(2)] + _S671[int(2)];
            float3  _S675 = _S603[int(3)] + _S671[int(3)];
            float3  _S676 = _S603[int(4)] + _S659;
            float3  _S677 = _S603[int(5)] + _S657;
            float3  _S678 = _S603[int(6)] + _S653;
            float3  _S679 = _S603[int(7)] + _S651;
            float3  _S680 = _S603[int(8)] + _S648;
            float3  _S681 = _S603[int(9)] + _S671[int(9)];
            float3  _S682 = _S603[int(10)] + _S671[int(10)];
            float3  _S683 = _S603[int(11)] + _S671[int(11)];
            float3  _S684 = _S603[int(12)] + _S671[int(12)];
            float3  _S685 = _S603[int(13)] + _S671[int(13)];
            float3  _S686 = _S603[int(14)] + _S671[int(14)];
            float3  _S687 = _S603[int(15)] + _S671[int(15)];
            float _S688 = _S662 + _S667 + _S670 + _S670 + _S520;
            float _S689 = _S664 + _S666 + _S669 + _S669 + _S521;
            _S514 = _S661 + _S663 + _S668 + _S668 + _S522;
            _S515 = _S689;
            _S516 = _S688;
            _S603[int(0)] = _S672;
            _S603[int(1)] = _S673;
            _S603[int(2)] = _S674;
            _S603[int(3)] = _S675;
            _S603[int(4)] = _S676;
            _S603[int(5)] = _S677;
            _S603[int(6)] = _S678;
            _S603[int(7)] = _S679;
            _S603[int(8)] = _S680;
            _S603[int(9)] = _S681;
            _S603[int(10)] = _S682;
            _S603[int(11)] = _S683;
            _S603[int(12)] = _S684;
            _S603[int(13)] = _S685;
            _S603[int(14)] = _S686;
            _S603[int(15)] = _S687;
        }
        else
        {
            _S514 = 0.0f;
            _S515 = 0.0f;
            _S516 = 0.0f;
            _S603[int(0)] = _S482;
            _S603[int(1)] = _S482;
            _S603[int(2)] = _S482;
            _S603[int(3)] = _S482;
            _S603[int(4)] = _S482;
            _S603[int(5)] = _S482;
            _S603[int(6)] = _S482;
            _S603[int(7)] = _S482;
            _S603[int(8)] = _S482;
            _S603[int(9)] = _S482;
            _S603[int(10)] = _S482;
            _S603[int(11)] = _S482;
            _S603[int(12)] = _S482;
            _S603[int(13)] = _S482;
            _S603[int(14)] = _S482;
            _S603[int(15)] = _S482;
        }
        float3  _S690 = - _S602.differential_0;
        float3  _S691 = _S508 * _S690;
        float3  _S692 = _S509 * _S690;
        float3  _S693 = _S510 * _S602.differential_0;
        float3  _S694 = _S511 * _S602.differential_0;
        float3  _S695 = _S512 * _S690;
        float3  _S696 = _S513 * _S690;
        float _S697 = 0.48860251903533936f * (_S694.x + _S694.y + _S694.z) + _S514;
        float _S698 = 0.48860251903533936f * (_S696.x + _S696.y + _S696.z) + _S515;
        float _S699 = 0.48860251903533936f * (_S692.x + _S692.y + _S692.z) + _S516;
        FixedArray<float3 , 16>  _S700;
        _S700[int(0)] = _S482;
        _S700[int(1)] = _S482;
        _S700[int(2)] = _S482;
        _S700[int(3)] = _S482;
        _S700[int(4)] = _S482;
        _S700[int(5)] = _S482;
        _S700[int(6)] = _S482;
        _S700[int(7)] = _S482;
        _S700[int(8)] = _S482;
        _S700[int(9)] = _S482;
        _S700[int(10)] = _S482;
        _S700[int(11)] = _S482;
        _S700[int(12)] = _S482;
        _S700[int(13)] = _S482;
        _S700[int(14)] = _S482;
        _S700[int(15)] = _S482;
        _S700[int(3)] = _S691;
        _S700[int(2)] = _S693;
        _S700[int(1)] = _S695;
        float3  _S701 = _S603[int(0)] + _S700[int(0)];
        float3  _S702 = _S603[int(1)] + _S695;
        float3  _S703 = _S603[int(2)] + _S693;
        float3  _S704 = _S603[int(3)] + _S691;
        float3  _S705 = _S603[int(4)] + _S700[int(4)];
        float3  _S706 = _S603[int(5)] + _S700[int(5)];
        float3  _S707 = _S603[int(6)] + _S700[int(6)];
        float3  _S708 = _S603[int(7)] + _S700[int(7)];
        float3  _S709 = _S603[int(8)] + _S700[int(8)];
        float3  _S710 = _S603[int(9)] + _S700[int(9)];
        float3  _S711 = _S603[int(10)] + _S700[int(10)];
        float3  _S712 = _S603[int(11)] + _S700[int(11)];
        float3  _S713 = _S603[int(12)] + _S700[int(12)];
        float3  _S714 = _S603[int(13)] + _S700[int(13)];
        float3  _S715 = _S603[int(14)] + _S700[int(14)];
        float3  _S716 = _S603[int(15)] + _S700[int(15)];
        features_12 = make_float3 (_S699, _S698, _S697);
        _S603[int(0)] = _S701;
        _S603[int(1)] = _S702;
        _S603[int(2)] = _S703;
        _S603[int(3)] = _S704;
        _S603[int(4)] = _S705;
        _S603[int(5)] = _S706;
        _S603[int(6)] = _S707;
        _S603[int(7)] = _S708;
        _S603[int(8)] = _S709;
        _S603[int(9)] = _S710;
        _S603[int(10)] = _S711;
        _S603[int(11)] = _S712;
        _S603[int(12)] = _S713;
        _S603[int(13)] = _S714;
        _S603[int(14)] = _S715;
        _S603[int(15)] = _S716;
    }
    else
    {
        features_12 = _S482;
        _S603[int(0)] = _S482;
        _S603[int(1)] = _S482;
        _S603[int(2)] = _S482;
        _S603[int(3)] = _S482;
        _S603[int(4)] = _S482;
        _S603[int(5)] = _S482;
        _S603[int(6)] = _S482;
        _S603[int(7)] = _S482;
        _S603[int(8)] = _S482;
        _S603[int(9)] = _S482;
        _S603[int(10)] = _S482;
        _S603[int(11)] = _S482;
        _S603[int(12)] = _S482;
        _S603[int(13)] = _S482;
        _S603[int(14)] = _S482;
        _S603[int(15)] = _S482;
    }
    float3  _S717 = make_float3 (0.282094806432724f) * _S602.differential_0;
    dpdirection_1->primal_0 = (*dpdirection_1).primal_0;
    dpdirection_1->differential_0 = features_12;
    FixedArray<float3 , 16>  _S718;
    _S718[int(0)] = _S482;
    _S718[int(1)] = _S482;
    _S718[int(2)] = _S482;
    _S718[int(3)] = _S482;
    _S718[int(4)] = _S482;
    _S718[int(5)] = _S482;
    _S718[int(6)] = _S482;
    _S718[int(7)] = _S482;
    _S718[int(8)] = _S482;
    _S718[int(9)] = _S482;
    _S718[int(10)] = _S482;
    _S718[int(11)] = _S482;
    _S718[int(12)] = _S482;
    _S718[int(13)] = _S482;
    _S718[int(14)] = _S482;
    _S718[int(15)] = _S482;
    _S718[int(0)] = _S717;
    FixedArray<float3 , 16>  _S719 = {
        _S603[int(0)] + _S717, _S603[int(1)] + _S718[int(1)], _S603[int(2)] + _S718[int(2)], _S603[int(3)] + _S718[int(3)], _S603[int(4)] + _S718[int(4)], _S603[int(5)] + _S718[int(5)], _S603[int(6)] + _S718[int(6)], _S603[int(7)] + _S718[int(7)], _S603[int(8)] + _S718[int(8)], _S603[int(9)] + _S718[int(9)], _S603[int(10)] + _S718[int(10)], _S603[int(11)] + _S718[int(11)], _S603[int(12)] + _S718[int(12)], _S603[int(13)] + _S718[int(13)], _S603[int(14)] + _S718[int(14)], _S603[int(15)] + _S718[int(15)]
    };
    dpcoefficients_1->primal_0 = dpcoefficients_1->primal_0;
    dpcoefficients_1->differential_0 = _S719;
    return;
}

__device__ void particleFeaturesIntegrateBwdToBuffer(float3  incidentDirection_2, float3  canonicalPosition_2, float3  * canonicalPositionGrad_0, float alpha_8, float * alphaGrad_3, uint particleIdx_9, float * featuresBufferPtr_0, float * featuresBufferGradPtr_0, int auxParam_2, bool exclusiveGradient_2, FixedArray<float, 3>  features_15, FixedArray<float, 3>  * integratedFeatures_3, FixedArray<float, 3>  * integratedFeaturesGrad_1)
{
    bool _S720 = alpha_8 > 0.0f;
    if(_S720)
    {
        shRadiativeParticle_ParametersBuffer_0 parametersBuffer_3;
        (&parametersBuffer_3)->_dataPtr_1 = (slang_bit_cast<float3  *>((slang_bit_cast<ulonglong>(featuresBufferPtr_0))));
        (&parametersBuffer_3)->_gradPtr_1 = (slang_bit_cast<float3  *>((slang_bit_cast<ulonglong>(featuresBufferGradPtr_0))));
        (&parametersBuffer_3)->exclusiveGradient_1 = exclusiveGradient_2;
        float _S721 = 1.0f / (1.0f - alpha_8);
        float _S722 = ((*integratedFeatures_3)[int(0)] - features_15[int(0)] * alpha_8) * _S721;
        (*integratedFeatures_3)[int(0)] = _S722;
        float _S723 = ((*integratedFeatures_3)[int(1)] - features_15[int(1)] * alpha_8) * _S721;
        (*integratedFeatures_3)[int(1)] = _S723;
        float _S724 = ((*integratedFeatures_3)[int(2)] - features_15[int(2)] * alpha_8) * _S721;
        (*integratedFeatures_3)[int(2)] = _S724;
        float3  integratedFeaturesVec_0;
        *&((&integratedFeaturesVec_0)->x) = _S722;
        float3  integratedFeaturesGradVec_0;
        *&((&integratedFeaturesGradVec_0)->x) = (*integratedFeaturesGrad_1)[int(0)];
        *&((&integratedFeaturesVec_0)->y) = _S723;
        *&((&integratedFeaturesGradVec_0)->y) = (*integratedFeaturesGrad_1)[int(1)];
        *&((&integratedFeaturesVec_0)->z) = _S724;
        *&((&integratedFeaturesGradVec_0)->z) = (*integratedFeaturesGrad_1)[int(2)];
        float3  _S725 = integratedFeaturesGradVec_0;
        uint _S726 = uint(auxParam_2);
        shRadiativeParticle_ParametersBuffer_0 _S727 = parametersBuffer_3;
        float3  _S728 = make_float3 (0.0f);
        FixedArray<float3 , 16>  _S729 = {
            _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728
        };
        shRadiativeParticle_Parameters_0 _S730 = { _S729 };
        s_bwd_prop_shRadiativeParticle_integrateRadianceFromBuffer_Intermediates_0 _S731;
        (&_S731)->_S467 = _S728;
        (&_S731)->_S468 = _S730;
        (&(&_S731)->_S468)->sphCoefficients_0 = _S729;
        (&_S731)->_S467 = integratedFeaturesVec_0;
        uint _S732 = particleIdx_9 * 16U;
        shRadiativeParticle_Parameters_0 parameters_5;
        (&parameters_5)->sphCoefficients_0[int(0)] = *(parametersBuffer_3._dataPtr_1 + _S732);
        uint _S733 = _S732 + 1U;
        (&parameters_5)->sphCoefficients_0[int(1)] = *(parametersBuffer_3._dataPtr_1 + _S733);
        uint _S734 = _S732 + 2U;
        (&parameters_5)->sphCoefficients_0[int(2)] = *(parametersBuffer_3._dataPtr_1 + _S734);
        uint _S735 = _S732 + 3U;
        (&parameters_5)->sphCoefficients_0[int(3)] = *(parametersBuffer_3._dataPtr_1 + _S735);
        uint _S736 = _S732 + 4U;
        (&parameters_5)->sphCoefficients_0[int(4)] = *(parametersBuffer_3._dataPtr_1 + _S736);
        uint _S737 = _S732 + 5U;
        (&parameters_5)->sphCoefficients_0[int(5)] = *(parametersBuffer_3._dataPtr_1 + _S737);
        uint _S738 = _S732 + 6U;
        (&parameters_5)->sphCoefficients_0[int(6)] = *(parametersBuffer_3._dataPtr_1 + _S738);
        uint _S739 = _S732 + 7U;
        (&parameters_5)->sphCoefficients_0[int(7)] = *(parametersBuffer_3._dataPtr_1 + _S739);
        uint _S740 = _S732 + 8U;
        (&parameters_5)->sphCoefficients_0[int(8)] = *(parametersBuffer_3._dataPtr_1 + _S740);
        uint _S741 = _S732 + 9U;
        (&parameters_5)->sphCoefficients_0[int(9)] = *(parametersBuffer_3._dataPtr_1 + _S741);
        uint _S742 = _S732 + 10U;
        (&parameters_5)->sphCoefficients_0[int(10)] = *(parametersBuffer_3._dataPtr_1 + _S742);
        uint _S743 = _S732 + 11U;
        (&parameters_5)->sphCoefficients_0[int(11)] = *(parametersBuffer_3._dataPtr_1 + _S743);
        uint _S744 = _S732 + 12U;
        (&parameters_5)->sphCoefficients_0[int(12)] = *(parametersBuffer_3._dataPtr_1 + _S744);
        uint _S745 = _S732 + 13U;
        (&parameters_5)->sphCoefficients_0[int(13)] = *(parametersBuffer_3._dataPtr_1 + _S745);
        uint _S746 = _S732 + 14U;
        (&parameters_5)->sphCoefficients_0[int(14)] = *(parametersBuffer_3._dataPtr_1 + _S746);
        uint _S747 = _S732 + 15U;
        (&parameters_5)->sphCoefficients_0[int(15)] = *(parametersBuffer_3._dataPtr_1 + _S747);
        (&_S731)->_S468 = parameters_5;
        float3  _S748;
        float3  dpintegratedRadiance_1;
        FixedArray<float3 , 16>  _S749;
        int _S750;
        if(_S720)
        {
            int _S751 = int(_S726);
            FixedArray<float3 , 16>  _S752 = (&(&_S731)->_S468)->sphCoefficients_0;
            FixedArray<float3 , 16>  _S753 = (&(&_S731)->_S468)->sphCoefficients_0;
            float3  _S754 = s_primal_ctx_sphericalHarmonics_decode_0(_S751, &_S753, incidentDirection_2);
            if(_S720)
            {
                _S748 = make_float3 (alpha_8);
            }
            else
            {
                _S748 = _S728;
            }
            float3  _S755 = _S748;
            _S748 = _S754;
            dpintegratedRadiance_1 = _S755;
            _S749 = _S752;
            _S750 = _S751;
        }
        else
        {
            _S748 = _S728;
            dpintegratedRadiance_1 = _S728;
            _S749[int(0)] = _S728;
            _S749[int(1)] = _S728;
            _S749[int(2)] = _S728;
            _S749[int(3)] = _S728;
            _S749[int(4)] = _S728;
            _S749[int(5)] = _S728;
            _S749[int(6)] = _S728;
            _S749[int(7)] = _S728;
            _S749[int(8)] = _S728;
            _S749[int(9)] = _S728;
            _S749[int(10)] = _S728;
            _S749[int(11)] = _S728;
            _S749[int(12)] = _S728;
            _S749[int(13)] = _S728;
            _S749[int(14)] = _S728;
            _S749[int(15)] = _S728;
            _S750 = int(0);
        }
        shRadiativeParticle_Parameters_0 _S756 = shRadiativeParticle_Parameters_x24_syn_dzero_0();
        float _S757;
        shRadiativeParticle_Parameters_0 _S758;
        if(_S720)
        {
            if(_S720)
            {
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S759;
                (&_S759)->primal_0 = (&_S731)->_S467;
                (&_S759)->differential_0 = _S728;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S760;
                (&_S760)->primal_0 = _S748;
                (&_S760)->differential_0 = _S728;
                DiffPair_vectorx3Cfloatx2C3x3E_0 _S761;
                (&_S761)->primal_0 = dpintegratedRadiance_1;
                (&_S761)->differential_0 = _S728;
                s_bwd_prop_lerp_0(&_S759, &_S760, &_S761, _S725);
                float _S762 = _S761.differential_0.x + _S761.differential_0.y + _S761.differential_0.z;
                _S748 = _S760.differential_0;
                dpintegratedRadiance_1 = _S759.differential_0;
                _S757 = _S762;
            }
            else
            {
                _S748 = _S728;
                dpintegratedRadiance_1 = _S725;
                _S757 = 0.0f;
            }
            FixedArray<float3 , 16>  _S763 = { _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728, _S728 };
            DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0 _S764;
            (&_S764)->primal_0 = _S749;
            (&_S764)->differential_0 = _S763;
            DiffPair_vectorx3Cfloatx2C3x3E_0 _S765;
            (&_S765)->primal_0 = incidentDirection_2;
            (&_S765)->differential_0 = _S728;
            s_bwd_prop_sphericalHarmonics_decode_0(_S750, &_S764, &_S765, _S748);
            shRadiativeParticle_Parameters_0 _S766 = _S756;
            (&_S766)->sphCoefficients_0 = (&_S764)->differential_0;
            shRadiativeParticle_Parameters_0 _S767 = _S756;
            shRadiativeParticle_Parameters_0 _S768 = _S766;
            shRadiativeParticle_Parameters_0 _S769 = shRadiativeParticle_Parameters_x24_syn_dadd_0(&_S767, &_S768);
            _S758 = _S769;
        }
        else
        {
            _S758 = _S756;
            dpintegratedRadiance_1 = _S725;
            _S757 = 0.0f;
        }
        shRadiativeParticle_Parameters_0 _S770 = _S758;
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S771 = _S727._gradPtr_1 + _S732;
            *&(_S771->x) = *&(_S771->x) + _S770.sphCoefficients_0[int(0)].x;
            *&(_S771->y) = *&(_S771->y) + _S770.sphCoefficients_0[int(0)].y;
            *&(_S771->z) = *&(_S771->z) + _S770.sphCoefficients_0[int(0)].z;
        }
        else
        {
            float3  * _S772 = _S727._gradPtr_1 + _S732;
            float _S773 = atomicAdd(&(_S772->x), _S770.sphCoefficients_0[int(0)].x);
            float _S774 = atomicAdd(&(_S772->y), _S770.sphCoefficients_0[int(0)].y);
            float _S775 = atomicAdd(&(_S772->z), _S770.sphCoefficients_0[int(0)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S776 = _S727._gradPtr_1 + _S733;
            *&(_S776->x) = *&(_S776->x) + _S770.sphCoefficients_0[int(1)].x;
            *&(_S776->y) = *&(_S776->y) + _S770.sphCoefficients_0[int(1)].y;
            *&(_S776->z) = *&(_S776->z) + _S770.sphCoefficients_0[int(1)].z;
        }
        else
        {
            float3  * _S777 = _S727._gradPtr_1 + _S733;
            float _S778 = atomicAdd(&(_S777->x), _S770.sphCoefficients_0[int(1)].x);
            float _S779 = atomicAdd(&(_S777->y), _S770.sphCoefficients_0[int(1)].y);
            float _S780 = atomicAdd(&(_S777->z), _S770.sphCoefficients_0[int(1)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S781 = _S727._gradPtr_1 + _S734;
            *&(_S781->x) = *&(_S781->x) + _S770.sphCoefficients_0[int(2)].x;
            *&(_S781->y) = *&(_S781->y) + _S770.sphCoefficients_0[int(2)].y;
            *&(_S781->z) = *&(_S781->z) + _S770.sphCoefficients_0[int(2)].z;
        }
        else
        {
            float3  * _S782 = _S727._gradPtr_1 + _S734;
            float _S783 = atomicAdd(&(_S782->x), _S770.sphCoefficients_0[int(2)].x);
            float _S784 = atomicAdd(&(_S782->y), _S770.sphCoefficients_0[int(2)].y);
            float _S785 = atomicAdd(&(_S782->z), _S770.sphCoefficients_0[int(2)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S786 = _S727._gradPtr_1 + _S735;
            *&(_S786->x) = *&(_S786->x) + _S770.sphCoefficients_0[int(3)].x;
            *&(_S786->y) = *&(_S786->y) + _S770.sphCoefficients_0[int(3)].y;
            *&(_S786->z) = *&(_S786->z) + _S770.sphCoefficients_0[int(3)].z;
        }
        else
        {
            float3  * _S787 = _S727._gradPtr_1 + _S735;
            float _S788 = atomicAdd(&(_S787->x), _S770.sphCoefficients_0[int(3)].x);
            float _S789 = atomicAdd(&(_S787->y), _S770.sphCoefficients_0[int(3)].y);
            float _S790 = atomicAdd(&(_S787->z), _S770.sphCoefficients_0[int(3)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S791 = _S727._gradPtr_1 + _S736;
            *&(_S791->x) = *&(_S791->x) + _S770.sphCoefficients_0[int(4)].x;
            *&(_S791->y) = *&(_S791->y) + _S770.sphCoefficients_0[int(4)].y;
            *&(_S791->z) = *&(_S791->z) + _S770.sphCoefficients_0[int(4)].z;
        }
        else
        {
            float3  * _S792 = _S727._gradPtr_1 + _S736;
            float _S793 = atomicAdd(&(_S792->x), _S770.sphCoefficients_0[int(4)].x);
            float _S794 = atomicAdd(&(_S792->y), _S770.sphCoefficients_0[int(4)].y);
            float _S795 = atomicAdd(&(_S792->z), _S770.sphCoefficients_0[int(4)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S796 = _S727._gradPtr_1 + _S737;
            *&(_S796->x) = *&(_S796->x) + _S770.sphCoefficients_0[int(5)].x;
            *&(_S796->y) = *&(_S796->y) + _S770.sphCoefficients_0[int(5)].y;
            *&(_S796->z) = *&(_S796->z) + _S770.sphCoefficients_0[int(5)].z;
        }
        else
        {
            float3  * _S797 = _S727._gradPtr_1 + _S737;
            float _S798 = atomicAdd(&(_S797->x), _S770.sphCoefficients_0[int(5)].x);
            float _S799 = atomicAdd(&(_S797->y), _S770.sphCoefficients_0[int(5)].y);
            float _S800 = atomicAdd(&(_S797->z), _S770.sphCoefficients_0[int(5)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S801 = _S727._gradPtr_1 + _S738;
            *&(_S801->x) = *&(_S801->x) + _S770.sphCoefficients_0[int(6)].x;
            *&(_S801->y) = *&(_S801->y) + _S770.sphCoefficients_0[int(6)].y;
            *&(_S801->z) = *&(_S801->z) + _S770.sphCoefficients_0[int(6)].z;
        }
        else
        {
            float3  * _S802 = _S727._gradPtr_1 + _S738;
            float _S803 = atomicAdd(&(_S802->x), _S770.sphCoefficients_0[int(6)].x);
            float _S804 = atomicAdd(&(_S802->y), _S770.sphCoefficients_0[int(6)].y);
            float _S805 = atomicAdd(&(_S802->z), _S770.sphCoefficients_0[int(6)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S806 = _S727._gradPtr_1 + _S739;
            *&(_S806->x) = *&(_S806->x) + _S770.sphCoefficients_0[int(7)].x;
            *&(_S806->y) = *&(_S806->y) + _S770.sphCoefficients_0[int(7)].y;
            *&(_S806->z) = *&(_S806->z) + _S770.sphCoefficients_0[int(7)].z;
        }
        else
        {
            float3  * _S807 = _S727._gradPtr_1 + _S739;
            float _S808 = atomicAdd(&(_S807->x), _S770.sphCoefficients_0[int(7)].x);
            float _S809 = atomicAdd(&(_S807->y), _S770.sphCoefficients_0[int(7)].y);
            float _S810 = atomicAdd(&(_S807->z), _S770.sphCoefficients_0[int(7)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S811 = _S727._gradPtr_1 + _S740;
            *&(_S811->x) = *&(_S811->x) + _S770.sphCoefficients_0[int(8)].x;
            *&(_S811->y) = *&(_S811->y) + _S770.sphCoefficients_0[int(8)].y;
            *&(_S811->z) = *&(_S811->z) + _S770.sphCoefficients_0[int(8)].z;
        }
        else
        {
            float3  * _S812 = _S727._gradPtr_1 + _S740;
            float _S813 = atomicAdd(&(_S812->x), _S770.sphCoefficients_0[int(8)].x);
            float _S814 = atomicAdd(&(_S812->y), _S770.sphCoefficients_0[int(8)].y);
            float _S815 = atomicAdd(&(_S812->z), _S770.sphCoefficients_0[int(8)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S816 = _S727._gradPtr_1 + _S741;
            *&(_S816->x) = *&(_S816->x) + _S770.sphCoefficients_0[int(9)].x;
            *&(_S816->y) = *&(_S816->y) + _S770.sphCoefficients_0[int(9)].y;
            *&(_S816->z) = *&(_S816->z) + _S770.sphCoefficients_0[int(9)].z;
        }
        else
        {
            float3  * _S817 = _S727._gradPtr_1 + _S741;
            float _S818 = atomicAdd(&(_S817->x), _S770.sphCoefficients_0[int(9)].x);
            float _S819 = atomicAdd(&(_S817->y), _S770.sphCoefficients_0[int(9)].y);
            float _S820 = atomicAdd(&(_S817->z), _S770.sphCoefficients_0[int(9)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S821 = _S727._gradPtr_1 + _S742;
            *&(_S821->x) = *&(_S821->x) + _S770.sphCoefficients_0[int(10)].x;
            *&(_S821->y) = *&(_S821->y) + _S770.sphCoefficients_0[int(10)].y;
            *&(_S821->z) = *&(_S821->z) + _S770.sphCoefficients_0[int(10)].z;
        }
        else
        {
            float3  * _S822 = _S727._gradPtr_1 + _S742;
            float _S823 = atomicAdd(&(_S822->x), _S770.sphCoefficients_0[int(10)].x);
            float _S824 = atomicAdd(&(_S822->y), _S770.sphCoefficients_0[int(10)].y);
            float _S825 = atomicAdd(&(_S822->z), _S770.sphCoefficients_0[int(10)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S826 = _S727._gradPtr_1 + _S743;
            *&(_S826->x) = *&(_S826->x) + _S770.sphCoefficients_0[int(11)].x;
            *&(_S826->y) = *&(_S826->y) + _S770.sphCoefficients_0[int(11)].y;
            *&(_S826->z) = *&(_S826->z) + _S770.sphCoefficients_0[int(11)].z;
        }
        else
        {
            float3  * _S827 = _S727._gradPtr_1 + _S743;
            float _S828 = atomicAdd(&(_S827->x), _S770.sphCoefficients_0[int(11)].x);
            float _S829 = atomicAdd(&(_S827->y), _S770.sphCoefficients_0[int(11)].y);
            float _S830 = atomicAdd(&(_S827->z), _S770.sphCoefficients_0[int(11)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S831 = _S727._gradPtr_1 + _S744;
            *&(_S831->x) = *&(_S831->x) + _S770.sphCoefficients_0[int(12)].x;
            *&(_S831->y) = *&(_S831->y) + _S770.sphCoefficients_0[int(12)].y;
            *&(_S831->z) = *&(_S831->z) + _S770.sphCoefficients_0[int(12)].z;
        }
        else
        {
            float3  * _S832 = _S727._gradPtr_1 + _S744;
            float _S833 = atomicAdd(&(_S832->x), _S770.sphCoefficients_0[int(12)].x);
            float _S834 = atomicAdd(&(_S832->y), _S770.sphCoefficients_0[int(12)].y);
            float _S835 = atomicAdd(&(_S832->z), _S770.sphCoefficients_0[int(12)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S836 = _S727._gradPtr_1 + _S745;
            *&(_S836->x) = *&(_S836->x) + _S770.sphCoefficients_0[int(13)].x;
            *&(_S836->y) = *&(_S836->y) + _S770.sphCoefficients_0[int(13)].y;
            *&(_S836->z) = *&(_S836->z) + _S770.sphCoefficients_0[int(13)].z;
        }
        else
        {
            float3  * _S837 = _S727._gradPtr_1 + _S745;
            float _S838 = atomicAdd(&(_S837->x), _S770.sphCoefficients_0[int(13)].x);
            float _S839 = atomicAdd(&(_S837->y), _S770.sphCoefficients_0[int(13)].y);
            float _S840 = atomicAdd(&(_S837->z), _S770.sphCoefficients_0[int(13)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S841 = _S727._gradPtr_1 + _S746;
            *&(_S841->x) = *&(_S841->x) + _S770.sphCoefficients_0[int(14)].x;
            *&(_S841->y) = *&(_S841->y) + _S770.sphCoefficients_0[int(14)].y;
            *&(_S841->z) = *&(_S841->z) + _S770.sphCoefficients_0[int(14)].z;
        }
        else
        {
            float3  * _S842 = _S727._gradPtr_1 + _S746;
            float _S843 = atomicAdd(&(_S842->x), _S770.sphCoefficients_0[int(14)].x);
            float _S844 = atomicAdd(&(_S842->y), _S770.sphCoefficients_0[int(14)].y);
            float _S845 = atomicAdd(&(_S842->z), _S770.sphCoefficients_0[int(14)].z);
        }
        if(_S727.exclusiveGradient_1)
        {
            float3  * _S846 = _S727._gradPtr_1 + _S747;
            *&(_S846->x) = *&(_S846->x) + _S770.sphCoefficients_0[int(15)].x;
            *&(_S846->y) = *&(_S846->y) + _S770.sphCoefficients_0[int(15)].y;
            *&(_S846->z) = *&(_S846->z) + _S770.sphCoefficients_0[int(15)].z;
        }
        else
        {
            float3  * _S847 = _S727._gradPtr_1 + _S747;
            float _S848 = atomicAdd(&(_S847->x), _S770.sphCoefficients_0[int(15)].x);
            float _S849 = atomicAdd(&(_S847->y), _S770.sphCoefficients_0[int(15)].y);
            float _S850 = atomicAdd(&(_S847->z), _S770.sphCoefficients_0[int(15)].z);
        }
        (*integratedFeaturesGrad_1)[int(0)] = dpintegratedRadiance_1.x;
        (*integratedFeaturesGrad_1)[int(1)] = dpintegratedRadiance_1.y;
        (*integratedFeaturesGrad_1)[int(2)] = dpintegratedRadiance_1.z;
        *alphaGrad_3 = *alphaGrad_3 + _S757;
    }
    return;
}

struct s_bwd_prop_shRadiativeParticle_radianceFromBuffer_Intermediates_0
{
    shRadiativeParticle_Parameters_0 _S851;
};

__device__ void particleFeaturesBwdToBuffer(uint particleIdx_10, float * featuresBufferPtr_1, float * featuresBufferGradPtr_1, int auxParam_3, bool exclusiveGradient_3, FixedArray<float, 3>  featuresGrad_1, float3  incidentDirection_3, float3  * incidentDirectionGrad_1)
{
    shRadiativeParticle_ParametersBuffer_0 parametersBuffer_4;
    (&parametersBuffer_4)->_dataPtr_1 = (slang_bit_cast<float3  *>((slang_bit_cast<ulonglong>(featuresBufferPtr_1))));
    (&parametersBuffer_4)->_gradPtr_1 = (slang_bit_cast<float3  *>((slang_bit_cast<ulonglong>(featuresBufferGradPtr_1))));
    (&parametersBuffer_4)->exclusiveGradient_1 = exclusiveGradient_3;
    float3  featuresGradVec_1;
    *&((&featuresGradVec_1)->x) = featuresGrad_1[int(0)];
    *&((&featuresGradVec_1)->y) = featuresGrad_1[int(1)];
    *&((&featuresGradVec_1)->z) = featuresGrad_1[int(2)];
    uint _S852 = uint(auxParam_3);
    shRadiativeParticle_ParametersBuffer_0 _S853 = parametersBuffer_4;
    float3  _S854 = make_float3 (0.0f);
    FixedArray<float3 , 16>  _S855 = {
        _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854
    };
    shRadiativeParticle_Parameters_0 _S856 = { _S855 };
    s_bwd_prop_shRadiativeParticle_radianceFromBuffer_Intermediates_0 _S857;
    (&_S857)->_S851 = _S856;
    (&(&_S857)->_S851)->sphCoefficients_0 = _S855;
    uint _S858 = particleIdx_10 * 16U;
    shRadiativeParticle_Parameters_0 parameters_6;
    (&parameters_6)->sphCoefficients_0[int(0)] = *(parametersBuffer_4._dataPtr_1 + _S858);
    uint _S859 = _S858 + 1U;
    (&parameters_6)->sphCoefficients_0[int(1)] = *(parametersBuffer_4._dataPtr_1 + _S859);
    uint _S860 = _S858 + 2U;
    (&parameters_6)->sphCoefficients_0[int(2)] = *(parametersBuffer_4._dataPtr_1 + _S860);
    uint _S861 = _S858 + 3U;
    (&parameters_6)->sphCoefficients_0[int(3)] = *(parametersBuffer_4._dataPtr_1 + _S861);
    uint _S862 = _S858 + 4U;
    (&parameters_6)->sphCoefficients_0[int(4)] = *(parametersBuffer_4._dataPtr_1 + _S862);
    uint _S863 = _S858 + 5U;
    (&parameters_6)->sphCoefficients_0[int(5)] = *(parametersBuffer_4._dataPtr_1 + _S863);
    uint _S864 = _S858 + 6U;
    (&parameters_6)->sphCoefficients_0[int(6)] = *(parametersBuffer_4._dataPtr_1 + _S864);
    uint _S865 = _S858 + 7U;
    (&parameters_6)->sphCoefficients_0[int(7)] = *(parametersBuffer_4._dataPtr_1 + _S865);
    uint _S866 = _S858 + 8U;
    (&parameters_6)->sphCoefficients_0[int(8)] = *(parametersBuffer_4._dataPtr_1 + _S866);
    uint _S867 = _S858 + 9U;
    (&parameters_6)->sphCoefficients_0[int(9)] = *(parametersBuffer_4._dataPtr_1 + _S867);
    uint _S868 = _S858 + 10U;
    (&parameters_6)->sphCoefficients_0[int(10)] = *(parametersBuffer_4._dataPtr_1 + _S868);
    uint _S869 = _S858 + 11U;
    (&parameters_6)->sphCoefficients_0[int(11)] = *(parametersBuffer_4._dataPtr_1 + _S869);
    uint _S870 = _S858 + 12U;
    (&parameters_6)->sphCoefficients_0[int(12)] = *(parametersBuffer_4._dataPtr_1 + _S870);
    uint _S871 = _S858 + 13U;
    (&parameters_6)->sphCoefficients_0[int(13)] = *(parametersBuffer_4._dataPtr_1 + _S871);
    uint _S872 = _S858 + 14U;
    (&parameters_6)->sphCoefficients_0[int(14)] = *(parametersBuffer_4._dataPtr_1 + _S872);
    uint _S873 = _S858 + 15U;
    (&parameters_6)->sphCoefficients_0[int(15)] = *(parametersBuffer_4._dataPtr_1 + _S873);
    (&_S857)->_S851 = parameters_6;
    int _S874 = int(_S852);
    FixedArray<float3 , 16>  _S875 = { _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854, _S854 };
    DiffPair_arrayx3Cvectorx3Cfloatx2C3x3Ex2C16x3E_0 _S876;
    (&_S876)->primal_0 = (&(&_S857)->_S851)->sphCoefficients_0;
    (&_S876)->differential_0 = _S875;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S877;
    (&_S877)->primal_0 = incidentDirection_3;
    (&_S877)->differential_0 = _S854;
    s_bwd_prop_sphericalHarmonics_decode_0(_S874, &_S876, &_S877, featuresGradVec_1);
    FixedArray<float3 , 16>  _S878 = (&_S876)->differential_0;
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S879 = _S877;
    if(parametersBuffer_4.exclusiveGradient_1)
    {
        float3  * _S880 = _S853._gradPtr_1 + _S858;
        *&(_S880->x) = *&(_S880->x) + _S878[int(0)].x;
        *&(_S880->y) = *&(_S880->y) + _S878[int(0)].y;
        *&(_S880->z) = *&(_S880->z) + _S878[int(0)].z;
    }
    else
    {
        float3  * _S881 = _S853._gradPtr_1 + _S858;
        float _S882 = atomicAdd(&(_S881->x), _S878[int(0)].x);
        float _S883 = atomicAdd(&(_S881->y), _S878[int(0)].y);
        float _S884 = atomicAdd(&(_S881->z), _S878[int(0)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S885 = _S853._gradPtr_1 + _S859;
        *&(_S885->x) = *&(_S885->x) + _S878[int(1)].x;
        *&(_S885->y) = *&(_S885->y) + _S878[int(1)].y;
        *&(_S885->z) = *&(_S885->z) + _S878[int(1)].z;
    }
    else
    {
        float3  * _S886 = _S853._gradPtr_1 + _S859;
        float _S887 = atomicAdd(&(_S886->x), _S878[int(1)].x);
        float _S888 = atomicAdd(&(_S886->y), _S878[int(1)].y);
        float _S889 = atomicAdd(&(_S886->z), _S878[int(1)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S890 = _S853._gradPtr_1 + _S860;
        *&(_S890->x) = *&(_S890->x) + _S878[int(2)].x;
        *&(_S890->y) = *&(_S890->y) + _S878[int(2)].y;
        *&(_S890->z) = *&(_S890->z) + _S878[int(2)].z;
    }
    else
    {
        float3  * _S891 = _S853._gradPtr_1 + _S860;
        float _S892 = atomicAdd(&(_S891->x), _S878[int(2)].x);
        float _S893 = atomicAdd(&(_S891->y), _S878[int(2)].y);
        float _S894 = atomicAdd(&(_S891->z), _S878[int(2)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S895 = _S853._gradPtr_1 + _S861;
        *&(_S895->x) = *&(_S895->x) + _S878[int(3)].x;
        *&(_S895->y) = *&(_S895->y) + _S878[int(3)].y;
        *&(_S895->z) = *&(_S895->z) + _S878[int(3)].z;
    }
    else
    {
        float3  * _S896 = _S853._gradPtr_1 + _S861;
        float _S897 = atomicAdd(&(_S896->x), _S878[int(3)].x);
        float _S898 = atomicAdd(&(_S896->y), _S878[int(3)].y);
        float _S899 = atomicAdd(&(_S896->z), _S878[int(3)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S900 = _S853._gradPtr_1 + _S862;
        *&(_S900->x) = *&(_S900->x) + _S878[int(4)].x;
        *&(_S900->y) = *&(_S900->y) + _S878[int(4)].y;
        *&(_S900->z) = *&(_S900->z) + _S878[int(4)].z;
    }
    else
    {
        float3  * _S901 = _S853._gradPtr_1 + _S862;
        float _S902 = atomicAdd(&(_S901->x), _S878[int(4)].x);
        float _S903 = atomicAdd(&(_S901->y), _S878[int(4)].y);
        float _S904 = atomicAdd(&(_S901->z), _S878[int(4)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S905 = _S853._gradPtr_1 + _S863;
        *&(_S905->x) = *&(_S905->x) + _S878[int(5)].x;
        *&(_S905->y) = *&(_S905->y) + _S878[int(5)].y;
        *&(_S905->z) = *&(_S905->z) + _S878[int(5)].z;
    }
    else
    {
        float3  * _S906 = _S853._gradPtr_1 + _S863;
        float _S907 = atomicAdd(&(_S906->x), _S878[int(5)].x);
        float _S908 = atomicAdd(&(_S906->y), _S878[int(5)].y);
        float _S909 = atomicAdd(&(_S906->z), _S878[int(5)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S910 = _S853._gradPtr_1 + _S864;
        *&(_S910->x) = *&(_S910->x) + _S878[int(6)].x;
        *&(_S910->y) = *&(_S910->y) + _S878[int(6)].y;
        *&(_S910->z) = *&(_S910->z) + _S878[int(6)].z;
    }
    else
    {
        float3  * _S911 = _S853._gradPtr_1 + _S864;
        float _S912 = atomicAdd(&(_S911->x), _S878[int(6)].x);
        float _S913 = atomicAdd(&(_S911->y), _S878[int(6)].y);
        float _S914 = atomicAdd(&(_S911->z), _S878[int(6)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S915 = _S853._gradPtr_1 + _S865;
        *&(_S915->x) = *&(_S915->x) + _S878[int(7)].x;
        *&(_S915->y) = *&(_S915->y) + _S878[int(7)].y;
        *&(_S915->z) = *&(_S915->z) + _S878[int(7)].z;
    }
    else
    {
        float3  * _S916 = _S853._gradPtr_1 + _S865;
        float _S917 = atomicAdd(&(_S916->x), _S878[int(7)].x);
        float _S918 = atomicAdd(&(_S916->y), _S878[int(7)].y);
        float _S919 = atomicAdd(&(_S916->z), _S878[int(7)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S920 = _S853._gradPtr_1 + _S866;
        *&(_S920->x) = *&(_S920->x) + _S878[int(8)].x;
        *&(_S920->y) = *&(_S920->y) + _S878[int(8)].y;
        *&(_S920->z) = *&(_S920->z) + _S878[int(8)].z;
    }
    else
    {
        float3  * _S921 = _S853._gradPtr_1 + _S866;
        float _S922 = atomicAdd(&(_S921->x), _S878[int(8)].x);
        float _S923 = atomicAdd(&(_S921->y), _S878[int(8)].y);
        float _S924 = atomicAdd(&(_S921->z), _S878[int(8)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S925 = _S853._gradPtr_1 + _S867;
        *&(_S925->x) = *&(_S925->x) + _S878[int(9)].x;
        *&(_S925->y) = *&(_S925->y) + _S878[int(9)].y;
        *&(_S925->z) = *&(_S925->z) + _S878[int(9)].z;
    }
    else
    {
        float3  * _S926 = _S853._gradPtr_1 + _S867;
        float _S927 = atomicAdd(&(_S926->x), _S878[int(9)].x);
        float _S928 = atomicAdd(&(_S926->y), _S878[int(9)].y);
        float _S929 = atomicAdd(&(_S926->z), _S878[int(9)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S930 = _S853._gradPtr_1 + _S868;
        *&(_S930->x) = *&(_S930->x) + _S878[int(10)].x;
        *&(_S930->y) = *&(_S930->y) + _S878[int(10)].y;
        *&(_S930->z) = *&(_S930->z) + _S878[int(10)].z;
    }
    else
    {
        float3  * _S931 = _S853._gradPtr_1 + _S868;
        float _S932 = atomicAdd(&(_S931->x), _S878[int(10)].x);
        float _S933 = atomicAdd(&(_S931->y), _S878[int(10)].y);
        float _S934 = atomicAdd(&(_S931->z), _S878[int(10)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S935 = _S853._gradPtr_1 + _S869;
        *&(_S935->x) = *&(_S935->x) + _S878[int(11)].x;
        *&(_S935->y) = *&(_S935->y) + _S878[int(11)].y;
        *&(_S935->z) = *&(_S935->z) + _S878[int(11)].z;
    }
    else
    {
        float3  * _S936 = _S853._gradPtr_1 + _S869;
        float _S937 = atomicAdd(&(_S936->x), _S878[int(11)].x);
        float _S938 = atomicAdd(&(_S936->y), _S878[int(11)].y);
        float _S939 = atomicAdd(&(_S936->z), _S878[int(11)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S940 = _S853._gradPtr_1 + _S870;
        *&(_S940->x) = *&(_S940->x) + _S878[int(12)].x;
        *&(_S940->y) = *&(_S940->y) + _S878[int(12)].y;
        *&(_S940->z) = *&(_S940->z) + _S878[int(12)].z;
    }
    else
    {
        float3  * _S941 = _S853._gradPtr_1 + _S870;
        float _S942 = atomicAdd(&(_S941->x), _S878[int(12)].x);
        float _S943 = atomicAdd(&(_S941->y), _S878[int(12)].y);
        float _S944 = atomicAdd(&(_S941->z), _S878[int(12)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S945 = _S853._gradPtr_1 + _S871;
        *&(_S945->x) = *&(_S945->x) + _S878[int(13)].x;
        *&(_S945->y) = *&(_S945->y) + _S878[int(13)].y;
        *&(_S945->z) = *&(_S945->z) + _S878[int(13)].z;
    }
    else
    {
        float3  * _S946 = _S853._gradPtr_1 + _S871;
        float _S947 = atomicAdd(&(_S946->x), _S878[int(13)].x);
        float _S948 = atomicAdd(&(_S946->y), _S878[int(13)].y);
        float _S949 = atomicAdd(&(_S946->z), _S878[int(13)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S950 = _S853._gradPtr_1 + _S872;
        *&(_S950->x) = *&(_S950->x) + _S878[int(14)].x;
        *&(_S950->y) = *&(_S950->y) + _S878[int(14)].y;
        *&(_S950->z) = *&(_S950->z) + _S878[int(14)].z;
    }
    else
    {
        float3  * _S951 = _S853._gradPtr_1 + _S872;
        float _S952 = atomicAdd(&(_S951->x), _S878[int(14)].x);
        float _S953 = atomicAdd(&(_S951->y), _S878[int(14)].y);
        float _S954 = atomicAdd(&(_S951->z), _S878[int(14)].z);
    }
    if(_S853.exclusiveGradient_1)
    {
        float3  * _S955 = _S853._gradPtr_1 + _S873;
        *&(_S955->x) = *&(_S955->x) + _S878[int(15)].x;
        *&(_S955->y) = *&(_S955->y) + _S878[int(15)].y;
        *&(_S955->z) = *&(_S955->z) + _S878[int(15)].z;
    }
    else
    {
        float3  * _S956 = _S853._gradPtr_1 + _S873;
        float _S957 = atomicAdd(&(_S956->x), _S878[int(15)].x);
        float _S958 = atomicAdd(&(_S956->y), _S878[int(15)].y);
        float _S959 = atomicAdd(&(_S956->z), _S878[int(15)].z);
    }
    *incidentDirectionGrad_1 = *incidentDirectionGrad_1 + _S879.differential_0;
    return;
}
