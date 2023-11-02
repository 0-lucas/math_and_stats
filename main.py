import math
from itertools import zip_longest


class Vector(object):
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError("The coordinates must be nonempty.")

        except TypeError:
            raise TypeError("The coordinates must be an iterable.")

    def __str__(self):
        return f"Vector : {self.coordinates}"

    def __eq__(self, other):
        return self.coordinates == other.coordinates

    def __add__(self, other):
        try:
            return Vector(list(
                x + y for x, y in zip(self.coordinates, other.coordinates)))

        except TypeError:
            raise TypeError("An object of incorrect type was passed. Please use a vector.")

        except ValueError:
            raise ValueError("The coordinates passed have different magnitudes.")

    def __sub__(self, other):
        try:
            return Vector(list((x - y for x, y in zip(self.coordinates, other.coordinates))))

        except TypeError:
            raise TypeError("An object of incorrect type was passed. Please use a vector.")

        except ValueError:
            raise ValueError("The coordinates passed have different magnitudes.")

    def __mul__(self, other):
        try:
            if isinstance(other, (float, int)):
                return Vector(list(i * other for i in self.coordinates))

            elif isinstance(other, Vector):
                if self.dimension == other.dimension:
                    return Vector(list(x * y for x, y in zip(self.coordinates, other.coordinates)))

                raise ValueError

            else:
                raise TypeError

        except TypeError:
            raise TypeError("An incorrect object type was passed. Please use a scalar")

        except ValueError:
            raise ValueError("The vector have different dimensions.")

    def is_zero_vector(self, tolerance=1e-10):
        return self.magnitude() < tolerance

    def magnitude(self):
        # Returns the vector magnitude.
        return math.sqrt(sum(coord ** 2 for coord in self.coordinates))

    def normalized(self):
        # Returns the normalized vector.
        try:
            return self.__mul__((1 / self.magnitude()))
        except ZeroDivisionError:
            raise ZeroDivisionError("The zero vector cannot be normalized.")

    def inner_product(self, other):
        # Returns the inner product of two vectors.
        try:
            if isinstance(other, Vector):
                return sum(x * y for x, y in zip_longest(self.coordinates, other.coordinates, fillvalue=0))

            raise TypeError

        except TypeError:
            raise TypeError("An incorrect object type was passed. Please use a vector.")

    def angle_with(self, other, in_rad=True):
        # Returns the smaller angle between two vectors.
        try:
            if isinstance(other, Vector):
                v1 = self.normalized()
                v2 = other.normalized()
                inner_product = round(v1.inner_product(v2), 5)
                angle = math.acos(inner_product)

                if in_rad:
                    # Output in radian.
                    return angle

                else:
                    # Output in degrees.
                    return math.degrees(angle)

            raise TypeError

        except TypeError:
            raise TypeError("An incorrect object type was passed. Please use a vector.")

    def is_orthogonal_with(self, other):
        return round(self.inner_product(other), 5) == 0

    def is_parallel_with(self, other):
        return (self.is_zero_vector() or
                other.is_zero_vector() or
                self.angle_with(other) == 0 or
                self.angle_with(other) == math.pi)


vec1 = Vector([-2.328, -7.284, -1.214])
vec2 = Vector([-1.821, 1.072, -2.94])

print(
    vec1.is_orthogonal_with(vec2)
)
