#pragma once

template <typename Number>
class Function1D
{
public:
  virtual Number
  value(const Number x) const = 0;
};

template <typename Number>
class Function1DConstant : public Function1D<Number>
{
public:
  Function1DConstant(const double y)
    : y(y)
  {}

  Number
  value(const Number x) const override
  {
    return y;
  }

private:
  double y;
};

template <typename Number>
class Function1DPiecewise : public Function1D<Number>
{
public:
  Function1DPiecewise(const bool extrapolate_linear = false)
    : extrapolate_linear(extrapolate_linear)
  {}

  Function1DPiecewise(const std::map<Number, Number> &pairs,
                      const bool extrapolate_linear = false)
    : pairs(pairs)
    , extrapolate_linear(extrapolate_linear)
  {}

  void
  add_pair(const Number x, const Number y)
  {
    pairs[x] = y;
  }

  Number
  value(const Number x) const override
  {
    if (pairs.empty())
      {
        return 0.;
      }
    else
      {
        typename std::map<Number, Number>::const_iterator it =
          std::find_if(pairs.begin(), pairs.end(), [x](const auto &val) {
            return val.first > x;
          });

        typename std::map<Number, Number>::const_iterator i1, i2;
        if (it == pairs.begin())
          {
            if (extrapolate_linear)
              {
                i1 = pairs.begin();
                i2 = pairs.begin();
                std::advance(i2, 1);
              }
            else
              {
                return it->second;
              }
          }
        else if (it == pairs.end())
          {
            if (extrapolate_linear)
              {
                i1 = pairs.end();
                std::advance(i1, -2);
                i2 = pairs.end();
                std::advance(i2, -1);
              }
            else
              {
                std::advance(it, -1);
                return it->second;
              }
          }
        else
          {
            i1 = it;
            std::advance(i1, -1);
            i2 = it;
          }

        const double k = (i1->second - i2->second) / (i1->first - i2->first);
        const double b = i1->second - k * i1->first;
        const double y = k * x + b;

        return y;
      }
  }

private:
  std::map<Number, Number> pairs;

  const bool extrapolate_linear;
};
