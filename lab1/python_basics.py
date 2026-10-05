"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value

    count = 0

    for char in text.lower():
        if char in "aeiou":
            count += 1
    return count

def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    seen = set()
    for char in text:
        if char in seen:
            return False
        seen.add(char)
    return True
    


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")



def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value

    steps = 0

    while number >= 10:
        product = 1

        for digit in str(number):
            product *= int(digit)

        number = product
        steps += 1

    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected

    total = 0

    for p, e in zip(predicted, expected):
        total += (p - e) ** 2

    return total / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    divisor = 2
    result = ""

    while number > 1:
        power = 0

        while number % divisor == 0:
            number //= divisor
            power += 1

        if power == 1:
            result += f"({divisor})"
        elif power > 1:
            result += f"({divisor}**{power})"

        divisor += 1

    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value

    total = 0
    k = 0

    while total < cube_count:
        k += 1
        total += k ** 2

    if total == cube_count:
        return k

    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = str(number)
    length = len(digits)

    if length % 2 == 1:
        middle = length // 2

        left = digits[:middle]
        right = digits[middle + 1:]

    else:
        middle = length // 2

        left = digits[:middle - 1]
        right = digits[middle + 1:]

    left_sum = sum(int(digit) for digit in left)
    right_sum = sum(int(digit) for digit in right)

    return left_sum == right_sum
