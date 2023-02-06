#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

/**
 * Code adopted from deal.II/includes/matrix_free/tensor_product_kernels.h for
 * tensors.
 */
template <int dim,
          int n_rows,
          int n_columns,
          typename Number,
          typename Number2,
          int  direction,
          bool contract_over_rows,
          bool add,
          int  type,
          bool one_line,
          int  n_components>
inline void
even_odd_apply(const Number2 *DEAL_II_RESTRICT                shapes,
               const dealii::Tensor<1, n_components, Number> *in,
               dealii::Tensor<1, n_components, Number> *      out)
{
  static_assert(type < 3, "Only three variants type=0,1,2 implemented");
  static_assert(one_line == false || direction == dim - 1,
                "Single-line evaluation only works for direction=dim-1.");


  Assert(dim == direction + 1 || one_line == true || n_rows == n_columns ||
           in != out,
         ExcMessage("In-place operation only supported for "
                    "n_rows==n_columns or single-line interpolation"));

  // We cannot statically assert that direction is less than dim, so must do
  // an additional dynamic check
  AssertIndexRange(direction, dim);

  const int nn     = contract_over_rows ? n_columns : n_rows;
  const int mm     = contract_over_rows ? n_rows : n_columns;
  const int n_cols = nn / 2;
  const int mid    = mm / 2;

  const int stride    = dealii::Utilities::pow(n_columns, direction);
  const int n_blocks1 = one_line ? 1 : stride;
  const int n_blocks2 =
    dealii::Utilities::pow(n_rows,
                           (direction >= dim) ? 0 : (dim - direction - 1));

  const int offset = (n_columns + 1) / 2;

  // this code may look very inefficient at first sight due to the many
  // different cases with if's at the innermost loop part, but all of the
  // conditionals can be evaluated at compile time because they are
  // templates, so the compiler should optimize everything away
  for (int i2 = 0; i2 < n_blocks2; ++i2)
    {
      for (int i1 = 0; i1 < n_blocks1; ++i1)
        {
          for (unsigned int c = 0; c < n_components; ++c)
            {
              Number xp[mid], xm[mid];
              for (int i = 0; i < mid; ++i)
                {
                  if (contract_over_rows == true && type == 1)
                    {
                      xp[i] = in[stride * i][c] - in[stride * (mm - 1 - i)][c];
                      xm[i] = in[stride * i][c] + in[stride * (mm - 1 - i)][c];
                    }
                  else
                    {
                      xp[i] = in[stride * i][c] + in[stride * (mm - 1 - i)][c];
                      xm[i] = in[stride * i][c] - in[stride * (mm - 1 - i)][c];
                    }
                }
              Number xmid = in[stride * mid][c];
              for (int col = 0; col < n_cols; ++col)
                {
                  Number r0, r1;
                  if (mid > 0)
                    {
                      if (contract_over_rows == true)
                        {
                          r0 = shapes[col] * xp[0];
                          r1 = shapes[(n_rows - 1) * offset + col] * xm[0];
                        }
                      else
                        {
                          r0 = shapes[col * offset] * xp[0];
                          r1 = shapes[(n_rows - 1 - col) * offset] * xm[0];
                        }
                      for (int ind = 1; ind < mid; ++ind)
                        {
                          if (contract_over_rows == true)
                            {
                              r0 += shapes[ind * offset + col] * xp[ind];
                              r1 += shapes[(n_rows - 1 - ind) * offset + col] *
                                    xm[ind];
                            }
                          else
                            {
                              r0 += shapes[col * offset + ind] * xp[ind];
                              r1 += shapes[(n_rows - 1 - col) * offset + ind] *
                                    xm[ind];
                            }
                        }
                    }
                  else
                    r0 = r1 = Number();
                  if (mm % 2 == 1 && contract_over_rows == true)
                    {
                      if (type == 1)
                        r1 += shapes[mid * offset + col] * xmid;
                      else
                        r0 += shapes[mid * offset + col] * xmid;
                    }
                  else if (mm % 2 == 1 && (nn % 2 == 0 || type > 0 || mm == 3))
                    r0 += shapes[col * offset + mid] * xmid;

                  if (add)
                    {
                      out[stride * col][c] += r0 + r1;
                      if (type == 1 && contract_over_rows == false)
                        out[stride * (nn - 1 - col)][c] += r1 - r0;
                      else
                        out[stride * (nn - 1 - col)][c] += r0 - r1;
                    }
                  else
                    {
                      out[stride * col][c] = r0 + r1;
                      if (type == 1 && contract_over_rows == false)
                        out[stride * (nn - 1 - col)][c] = r1 - r0;
                      else
                        out[stride * (nn - 1 - col)][c] = r0 - r1;
                    }
                }
              if (type == 0 && contract_over_rows == true && nn % 2 == 1 &&
                  mm % 2 == 1 && mm > 3)
                {
                  if (add)
                    out[stride * n_cols][c] +=
                      shapes[mid * offset + n_cols] * xmid;
                  else
                    out[stride * n_cols][c] =
                      shapes[mid * offset + n_cols] * xmid;
                }
              else if (contract_over_rows == true && nn % 2 == 1)
                {
                  Number r0;
                  if (mid > 0)
                    {
                      r0 = shapes[n_cols] * xp[0];
                      for (int ind = 1; ind < mid; ++ind)
                        r0 += shapes[ind * offset + n_cols] * xp[ind];
                    }
                  else
                    r0 = Number();
                  if (type != 1 && mm % 2 == 1)
                    r0 += shapes[mid * offset + n_cols] * xmid;

                  if (add)
                    out[stride * n_cols][c] += r0;
                  else
                    out[stride * n_cols][c] = r0;
                }
              else if (contract_over_rows == false && nn % 2 == 1)
                {
                  Number r0;
                  if (mid > 0)
                    {
                      if (type == 1)
                        {
                          r0 = shapes[n_cols * offset] * xm[0];
                          for (int ind = 1; ind < mid; ++ind)
                            r0 += shapes[n_cols * offset + ind] * xm[ind];
                        }
                      else
                        {
                          r0 = shapes[n_cols * offset] * xp[0];
                          for (int ind = 1; ind < mid; ++ind)
                            r0 += shapes[n_cols * offset + ind] * xp[ind];
                        }
                    }
                  else
                    r0 = Number();

                  if ((type == 0 || type == 2) && mm % 2 == 1)
                    r0 += shapes[n_cols * offset + mid] * xmid;

                  if (add)
                    out[stride * n_cols][c] += r0;
                  else
                    out[stride * n_cols][c] = r0;
                }
            }

          if (one_line == false)
            {
              in += 1;
              out += 1;
            }
        }
      if (one_line == false)
        {
          in += stride * (mm - 1);
          out += stride * (nn - 1);
        }
    }
}
