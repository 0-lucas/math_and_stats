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

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return self.__mul__(other)

    def __round__(self, n=3):
        return Vector(list(round(i, n) for i in self.coordinates))

    def is_zero_vector(self, tolerance=1e-10):
        return self.magnitude() < tolerance

    def magnitude(self):
        # Returns the vector magnitude.
        return math.sqrt(sum(i ** 2 for i in self.coordinates))

    def normalized(self):
        # Returns the normalized vector.
        try:
            return self.__mul__((1 / self.magnitude()))
        except ZeroDivisionError:
            raise ZeroDivisionError("The zero vector cannot be normalized.")

    def product(self, other):
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
                inner_product = round(v1.product(v2), 5)
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
        # Returns True if self is orthogonal with the passed vector, False if not.
        return round(self.product(other), 5) == 0

    def is_parallel_with(self, other):
        # Returns True if self is parallel with the passed vector, False if not.
        return (self.is_zero_vector() or
                other.is_zero_vector() or
                self.angle_with(other) == 0 or
                self.angle_with(other) == math.pi)

    def projected_in(self, other):
        # Returns the projection of self into the passed vector (parallel component).
        return self.product(other.normalized()) * other.normalized()

    def orthogonal_component_with(self, other):
        # Returns the orthogonal component of self referencing the passed vector.
        return self - self.projected_in(other)

    def cross_product(self, other):
        # Returns the cross product between two vectors.
        try:
            x1, y1, z1 = self.coordinates
            x2, y2, z2 = other.coordinates

            cross_vector = Vector([
                y1*z2 - y2*z1,
                -(x1*z2 - x2*z1),
                x1*y2 - x2*y1
            ])

            return cross_vector

        except ValueError as e:
            if str(e) == "not enough values to unpack (expected 3, got 2)":
                self_3d = Vector((*self.coordinates, 0))
                other_3d = Vector((*other.coordinates, 0))
                return self_3d.cross_product(other_3d)

            raise ValueError("The vectors should be 2D or 3D.")

        except TypeError:
            raise TypeError("An incorrect object type was passed. Please use a vector.")

    def parallelogram_area_with(self, other):
        # Returns the area of the parallelogram built by projection of the sum of vectors.
        return self.cross_product(other).magnitude()

    def triangle_area_with(self, other):
        # Returns the area of the triangle between the projection of two vectors and its connection.
        return self.parallelogram_area_with(other) / 2


vec1 = Vector([1.5, 9.547, 3.691])
vec2 = Vector([-6.007, 0.124, 5.772])

print(
    vec1.triangle_area_with(vec2).__round__(3)
)
