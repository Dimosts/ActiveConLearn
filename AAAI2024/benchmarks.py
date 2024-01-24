import random
import cpmpy as cp
from cpmpy import *
from utils import *
from cpmpy.transformations.normalize import toplevel_list


# Benchmark construction
def construct_4sudoku():
    # Variables
    grid = intvar(1, 4, shape=(4, 4), name="grid")

    model = Model()

    # Constraints on rows and columns
    for row in grid:
        model += AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += AllDifferent(col).decompose()

    # Constraints on blocks
    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            model += AllDifferent(grid[i:i + 2, j:j + 2]).decompose()  # python's indexing

    C = list(model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    C_T = set(toplevel_list(C))


    return grid, C_T, model


def construct_9sudoku():
    # Variables
    grid = intvar(1, 9, shape=(9, 9), name="grid")

    model = Model()

    # Constraints on rows and columns
    for row in grid:
        model += AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += AllDifferent(col).decompose()

    # Constraints on blocks
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            model += AllDifferent(grid[i:i + 3, j:j + 3]).decompose()  # python's indexing

    C = list( model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    C_T = set(toplevel_list(C))

    print(len(C_T))

    return grid, C_T, model

def construct_nurse_rostering(num_nurses, shifts_per_day, num_days):

    # Define the variables
    roster_matrix = intvar(1, num_nurses, shape=(num_days, shifts_per_day), name="shifts")
    print(roster_matrix)

    # Define the constraints
    model = Model()

    # Constraint: Each shift in a day must be assigned to a different nurse
    for day in range(num_days):
        model += AllDifferent(roster_matrix[day,:]).decompose()

    # Constraint: The last shift of a day cannot have the same nurse as the first shift of the next day
    for day in range(num_days - 1):
        model += (roster_matrix[day, shifts_per_day - 1] != roster_matrix[day + 1, 0])

    print(model)

    if model.solve():
        print(roster_matrix.value())
    else:
        print("no solution")
        raise Exception("The problem has no solution")

    C = list(model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    C_T = set(toplevel_list(C))

    return roster_matrix, C_T, model

def construct_nurse_rostering_advanced(num_nurses, shifts_per_day, nurses_per_shift, num_days):

    # Define the variables
    roster_matrix = intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="shifts")

    # Define the constraints
    model = Model()

    # Constraint: Each shift in a day must be assigned to a different nurse
    for day in range(num_days):
        model += AllDifferent(roster_matrix[day,...]).decompose()

    # Constraint: The last shift of a day cannot have the same nurse as the first shift of the next day
    for day in range(num_days - 1):
        model += AllDifferent(roster_matrix[day, shifts_per_day - 1], roster_matrix[day + 1, 0]).decompose()

    if model.solve():
        print("solution exists")
    else:
        print("no solution")
        raise Exception("The problem has no solution")

    C = list(model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    C_T = set(toplevel_list(C))

    return roster_matrix, C_T, model


def construct_examtt_simple(NSemesters=9, courses_per_semester=6, rooms=3, timeslots_per_day=3, days_for_exams=14):

    total_courses = NSemesters * courses_per_semester
    slots_per_day = rooms * timeslots_per_day
    total_slots = slots_per_day * days_for_exams

    # Variables
    courses = intvar(1, total_slots, shape=(NSemesters, courses_per_semester), name="courses")
    all_courses = set(courses.flatten())

    model = Model()

    model += AllDifferent(all_courses).decompose()

    # Constraints on courses of same semester
    for row in courses:
        model += AllDifferent(row // slots_per_day).decompose()

    C = list(model.constraints)

    print(model)

    if model.solve():
        print(courses.value())
    else:
        print("no solution")

    C2 = [c for c in model.constraints if not isinstance(c, list)]

    # it is needed to be able to make the "suboracle" in ask_query
    C_T = set(toplevel_list(C))

    print("new model: ----------------------------------\n", C_T)
    print(C2)

    return courses, C_T, model


def construct_examtt_advanced(NSemesters=9, courses_per_semester=6, rooms=3, timeslots_per_day=3, days_for_exams=14,
                            NProfessors=30):
    total_courses = NSemesters * courses_per_semester
    slots_per_day = rooms * timeslots_per_day
    total_slots = slots_per_day * days_for_exams

    # Variables
    courses = intvar(1, total_slots, shape=(NSemesters, courses_per_semester), name="courses")
    all_courses = set(courses.flatten())

    model = Model()

    model += AllDifferent(all_courses).decompose()

    # Constraints on courses of same semester
    for row in courses:
        model += AllDifferent(row // slots_per_day).decompose()

    C = list(model.constraints)

    # Constraints of Professors - instance specific -------------------------------

    # first define the courses each professor is assigned to
    # this can be given, or random generated!!

    assert NProfessors <= total_courses
    courses_per_professor = total_courses // NProfessors
    remaining_courses = total_courses % NProfessors  # will assign 1 per professor to some professors

    # probabilities of additional constraints to be introduced
    pcon_close = 0.3  # probability of professor constraint to have his courses on close days
    # (e.g. because he lives in another city and has to come for the exams)

    #pcon_diff = 0.2  # probability of professor constraint to not have his exams in a certain day

    Prof_courses = list()
    for i in range(NProfessors):

        prof_courses = list()

        for j in range(courses_per_professor):  # assign the calculated number of courses to the professors
            prof_courses.append(all_courses.pop())  # it is a set, so it will pop a random one (there is no order)

        if i < remaining_courses:  # # assign the remaining courses to the professors
            prof_courses.append(all_courses.pop())  # it is a set, so it will pop a random one (there is no order)

        Prof_courses.append(prof_courses)

        if len(prof_courses) > 1:

            r = random.uniform(0, 1)

            if r < pcon_close:
                for c1, c2 in all_pairs(prof_courses):
                    model += abs(c1 - c2) // slots_per_day <= 2  # all her courses in 2 days

    print(model)

    if model.solve():
        print(courses.value())
    else:
        print("no solution")

    C2 = [c for c in model.constraints if not isinstance(c, list)]

    # it is needed to be able to make the "suboracle" in ask_query
    C_T = set(toplevel_list(C))


    print("new model: ----------------------------------\n", C_T)
    print(C2)

    return courses, C_T, model


def construct_jsudoku():
    # Variables
    grid = intvar(1, 9, shape=(9, 9), name="grid")

    model = Model()

    # Constraints on rows and columns
    for row in grid:
        model += AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += AllDifferent(col).decompose()

    # the 9 blocks of squares in the specific instance of jsudoku
    blocks = [
        [grid[0, 0], grid[0, 1], grid[0, 2], grid[1, 0], grid[1, 1], grid[1, 2], grid[2, 0], grid[2, 1], grid[2, 2]],
        [grid[0, 3], grid[0, 4], grid[0, 5], grid[0, 6], grid[1, 3], grid[1, 4], grid[1, 5], grid[1, 6], grid[2, 4]],
        [grid[0, 7], grid[0, 8], grid[1, 7], grid[1, 8], grid[2, 8], grid[3, 8], grid[4, 8], grid[5, 7], grid[5, 8]],
        [grid[4, 1], grid[4, 2], grid[4, 3], grid[5, 1], grid[5, 2], grid[5, 3], grid[6, 1], grid[6, 2], grid[6, 3]],
        [grid[2, 3], grid[3, 2], grid[3, 3], grid[3, 4], grid[4, 4], grid[5, 4], grid[5, 5], grid[5, 6], grid[6, 5]],
        [grid[2, 5], grid[2, 6], grid[2, 7], grid[3, 5], grid[3, 6], grid[3, 7], grid[4, 5], grid[4, 6], grid[4, 7]],
        [grid[3, 0], grid[3, 1], grid[4, 0], grid[5, 0], grid[6, 0], grid[7, 0], grid[7, 1], grid[8, 0], grid[8, 1]],
        [grid[6, 4], grid[7, 2], grid[7, 3], grid[7, 4], grid[7, 5], grid[8, 2], grid[8, 3], grid[8, 4], grid[8, 5]],
        [grid[6, 6], grid[6, 7], grid[6, 8], grid[7, 6], grid[7, 7], grid[7, 8], grid[8, 6], grid[8, 7], grid[8, 8]]
    ]

    # Constraints on blocks
    for i in range(0, 9):
        model += AllDifferent(blocks[i][:]).decompose()  # python's indexing

    # it is needed to be able to make the "suboracle" in ask_query
    C_T = set(toplevel_list(model.constraints))


    print(len(C_T))

    return grid, C_T, model

def construct_new_random():

    model = Model()
    grid = intvar(1, 5, shape=(1, 100), name="grid")

    constraints = grid[0, 12] <= grid[0, 74], grid[0, 18] != grid[0, 82], grid[0, 55] >= grid[0, 79], grid[0, 43] > \
                  grid[0, 87], grid[0, 39] < grid[0, 62], grid[0, 68] != grid[0, 78], grid[0, 21] > grid[0, 83], grid[
                      0, 89] > grid[0, 95], grid[0, 21] != grid[0, 88], grid[0, 12] < grid[0, 41], grid[0, 13] != grid[
                      0, 90], grid[0, 64] < grid[0, 79], grid[0, 22] < grid[0, 86], grid[0, 42] <= grid[0, 62], grid[
                      0, 8] == grid[0, 53], grid[0, 68] >= grid[0, 99], grid[0, 21] == grid[0, 45], grid[0, 68] <= grid[
                      0, 86], grid[0, 26] != grid[0, 80], grid[0, 5] != grid[0, 93], grid[0, 49] == grid[0, 83], grid[
                      0, 33] <= grid[0, 58], grid[0, 35] != grid[0, 48], grid[0, 37] > grid[0, 95], grid[0, 37] != grid[
                      0, 89], grid[0, 2] < grid[0, 10], grid[0, 57] > grid[0, 71], grid[0, 8] <= grid[0, 83], grid[
                      0, 41] != grid[0, 85], grid[0, 86] != grid[0, 88], grid[0, 65] > grid[0, 91], grid[0, 74] >= grid[
                      0, 89], grid[0, 10] != grid[0, 81], grid[0, 47] != grid[0, 59], grid[0, 14] <= grid[0, 63], grid[
                      0, 61] > grid[0, 65], grid[0, 42] > grid[0, 98], grid[0, 75] > grid[0, 94], grid[0, 4] <= grid[
                      0, 41], grid[0, 50] != grid[0, 59], grid[0, 89] != grid[0, 97], grid[0, 57] >= grid[0, 97], grid[
                      0, 81] == grid[0, 90], grid[0, 37] <= grid[0, 70], grid[0, 25] > grid[0, 42], grid[0, 38] >= grid[
                      0, 90], grid[0, 34] <= grid[0, 94], grid[0, 38] != grid[0, 79], grid[0, 2] != grid[0, 54], grid[
                      0, 59] < grid[0, 86], grid[0, 4] < grid[0, 39], grid[0, 9] > grid[0, 93], grid[0, 34] <= grid[
                      0, 65], grid[0, 45] < grid[0, 48], grid[0, 21] <= grid[0, 37], grid[0, 34] == grid[0, 97], grid[
                      0, 12] == grid[0, 28], grid[0, 18] > grid[0, 43], grid[0, 6] >= grid[0, 58], grid[0, 6] == grid[
                      0, 31], grid[0, 31] < grid[0, 41], grid[0, 4] < grid[0, 34], grid[0, 31] >= grid[0, 37], grid[
                      0, 7] > grid[0, 8], grid[0, 68] == grid[0, 81], grid[0, 7] < grid[0, 48], grid[0, 47] != grid[
                      0, 56], grid[0, 41] <= grid[0, 77], grid[0, 3] > grid[0, 95], grid[0, 0] <= grid[0, 77], grid[
                      0, 43] == grid[0, 97], grid[0, 31] < grid[0, 75], grid[0, 35] > grid[0, 97], grid[0, 46] < grid[
                      0, 95], grid[0, 11] < grid[0, 78], grid[0, 16] != grid[0, 40], grid[0, 1] > grid[0, 45], grid[
                      0, 26] > grid[0, 69], grid[0, 9] != grid[0, 16], grid[0, 33] > grid[0, 72], grid[0, 28] < grid[
                      0, 36], grid[0, 50] < grid[0, 54], grid[0, 23] >= grid[0, 58], grid[0, 70] > grid[0, 88], grid[
                      0, 52] < grid[0, 70], grid[0, 51] < grid[0, 60], grid[0, 26] != grid[0, 46], grid[0, 3] != grid[
                      0, 99], grid[0, 55] >= grid[0, 72], grid[0, 3] == grid[0, 97], grid[0, 14] == grid[0, 69], grid[
                      0, 16] > grid[0, 76], grid[0, 53] < grid[0, 62], grid[0, 2] < grid[0, 67], grid[0, 53] > grid[
                      0, 68], grid[0, 40] < grid[0, 72], grid[0, 25] <= grid[0, 41], grid[0, 11] > grid[0, 95], grid[
                      0, 50] > grid[0, 65], grid[0, 3] == grid[0, 44], grid[0, 63] < grid[0, 88], grid[0, 29] < grid[
                      0, 70], grid[0, 85] == grid[0, 86], grid[0, 21] < grid[0, 85], grid[0, 5] >= grid[0, 68], grid[
                      0, 22] <= grid[0, 43], grid[0, 6] != grid[0, 55], grid[0, 22] <= grid[0, 41], grid[0, 67] >= grid[
                      0, 73], grid[0, 20] > grid[0, 78], grid[0, 27] == grid[0, 59], grid[0, 12] < grid[0, 92], grid[
                      0, 48] >= grid[0, 65], grid[0, 15] > grid[0, 58], grid[0, 2] > grid[0, 91], grid[0, 34] != grid[
                      0, 58], grid[0, 21] != grid[0, 57], grid[0, 24] == grid[0, 37], grid[0, 6] > grid[0, 24], grid[
                      0, 4] < grid[0, 42], grid[0, 25] <= grid[0, 96], grid[0, 22] <= grid[0, 60], grid[0, 5] < grid[
                      0, 11], grid[0, 12] <= grid[0, 94], grid[0, 27] <= grid[0, 58], grid[0, 25] != grid[0, 92], grid[
                      0, 32] >= grid[0, 59], grid[0, 24] <= grid[0, 25], grid[0, 42] <= grid[0, 67], grid[0, 55] != \
                  grid[0, 97], grid[0, 65] < grid[0, 96], grid[0, 6] != grid[0, 52], grid[0, 22] > grid[0, 71], grid[
                      0, 32] == grid[0, 86], grid[0, 29] > grid[0, 37], grid[0, 27] <= grid[0, 98], grid[0, 19] != grid[
                      0, 64], grid[0, 14] != grid[0, 43], grid[0, 43] != grid[0, 72], grid[0, 2] >= grid[0, 21], grid[
                      0, 9] >= grid[0, 10], grid[0, 28] <= grid[0, 70], grid[0, 1] <= grid[0, 93], grid[0, 39] >= grid[
                      0, 76], grid[0, 23] >= grid[0, 68], grid[0, 34] >= grid[0, 49], grid[0, 16] > grid[0, 98], grid[
                      0, 3] != grid[0, 42], grid[0, 67] < grid[0, 92], grid[0, 57] != grid[0, 89], grid[0, 39] >= grid[
                      0, 74], grid[0, 69] != grid[0, 93], grid[0, 57] >= grid[0, 65], grid[0, 53] <= grid[0, 96], grid[
                      0, 45] <= grid[0, 93], grid[0, 40] != grid[0, 60], grid[0, 38] >= grid[0, 58], grid[0, 1] > grid[
                      0, 51], grid[0, 29] <= grid[0, 31], grid[0, 74] != grid[0, 81], grid[0, 21] != grid[0, 72], grid[
                      0, 5] <= grid[0, 13], grid[0, 15] <= grid[0, 36], grid[0, 24] < grid[0, 35], grid[0, 48] > grid[
                      0, 73], grid[0, 54] != grid[0, 67], grid[0, 73] != grid[0, 98], grid[0, 13] >= grid[0, 30], grid[
                      0, 82] != grid[0, 93], grid[0, 60] <= grid[0, 70], grid[0, 27] < grid[0, 62], grid[0, 9] >= grid[
                      0, 84], grid[0, 36] <= grid[0, 66], grid[0, 47] >= grid[0, 90], grid[0, 35] < grid[0, 92], grid[
                      0, 20] >= grid[0, 80], grid[0, 44] < grid[0, 62], grid[0, 46] != grid[0, 93], grid[0, 0] >= grid[
                      0, 66], grid[0, 72] != grid[0, 87], grid[0, 41] > grid[0, 78], grid[0, 26] != grid[0, 40], grid[
                      0, 66] != grid[0, 78], grid[0, 18] > grid[0, 61], grid[0, 13] == grid[0, 95], grid[0, 90] <= grid[
                      0, 96], grid[0, 16] <= grid[0, 75], grid[0, 46] != grid[0, 51], grid[0, 25] > grid[0, 45], grid[
                      0, 21] > grid[0, 30], grid[0, 37] <= grid[0, 58], grid[0, 4] < grid[0, 36], grid[0, 15] >= grid[
                      0, 53], grid[0, 1] >= grid[0, 5], grid[0, 2] >= grid[0, 69], grid[0, 6] > grid[0, 40], grid[
                      0, 17] != grid[0, 25], grid[0, 26] > grid[0, 94], grid[0, 23] == grid[0, 35], grid[0, 2] >= grid[
                      0, 84]


    model += constraints
    C = list(model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query
    temp = []
    [temp.extend(c) for c in C]
    C_T = set(temp)

    return grid, C_T, model

def construct_random122():
    # Variables
    grid = intvar(1, 10, shape=(1, 50), name="grid")

    model = Model()

    # 122 constraints randomly generated
    constraints = [{grid[0, 11], grid[0, 23]}, {grid[0, 4], grid[0, 35]}, {grid[0, 15], grid[0, 23]},
                   {grid[0, 18], grid[0, 29]},
                   {grid[0, 22], grid[0, 32]}, {grid[0, 16], grid[0, 32]}, {grid[0, 22], grid[0, 43]},
                   {grid[0, 24], grid[0, 37]},
                   {grid[0, 20], grid[0, 43]}, {grid[0, 26], grid[0, 31]}, {grid[0, 36], grid[0, 48]},
                   {grid[0, 12], grid[0, 18]},
                   {grid[0, 3], grid[0, 12]}, {grid[0, 14], grid[0, 34]}, {grid[0, 24], grid[0, 41]},
                   {grid[0, 16], grid[0, 49]},
                   {grid[0, 43], grid[0, 49]}, {grid[0, 22], grid[0, 42]}, {grid[0, 6], grid[0, 30]},
                   {grid[0, 20], grid[0, 36]},
                   {grid[0, 32], grid[0, 40]}, {grid[0, 7], grid[0, 11]}, {grid[0, 5], grid[0, 45]},
                   {grid[0, 24], grid[0, 48]},
                   {grid[0, 26], grid[0, 49]}, {grid[0, 3], grid[0, 21]}, {grid[0, 11], grid[0, 39]},
                   {grid[0, 7], grid[0, 41]},
                   {grid[0, 3], grid[0, 49]}, {grid[0, 23], grid[0, 43]}, {grid[0, 39], grid[0, 42]},
                   {grid[0, 28], grid[0, 39]},
                   {grid[0, 2], grid[0, 6]}, {grid[0, 31], grid[0, 38]}, {grid[0, 20], grid[0, 38]},
                   {grid[0, 3], grid[0, 36]},
                   {grid[0, 11], grid[0, 45]}, {grid[0, 13], grid[0, 35]}, {grid[0, 2], grid[0, 27]},
                   {grid[0, 41], grid[0, 43]},
                   {grid[0, 31], grid[0, 36]}, {grid[0, 7], grid[0, 35]}, {grid[0, 32], grid[0, 48]},
                   {grid[0, 7], grid[0, 43]},
                   {grid[0, 1], grid[0, 14]}, {grid[0, 20], grid[0, 22]}, {grid[0, 12], grid[0, 34]},
                   {grid[0, 3], grid[0, 10]},
                   {grid[0, 30], grid[0, 49]}, {grid[0, 18], grid[0, 47]}, {grid[0, 15], grid[0, 18]},
                   {grid[0, 10], grid[0, 21]},
                   {grid[0, 7], grid[0, 49]}, {grid[0, 28], grid[0, 41]}, {grid[0, 2], grid[0, 35]},
                   {grid[0, 31], grid[0, 40]},
                   {grid[0, 38], grid[0, 49]}, {grid[0, 9], grid[0, 28]}, {grid[0, 2], grid[0, 15]},
                   {grid[0, 38], grid[0, 42]},
                   {grid[0, 29], grid[0, 30]}, {grid[0, 30], grid[0, 33]}, {grid[0, 16], grid[0, 38]},
                   {grid[0, 16], grid[0, 41]},
                   {grid[0, 37], grid[0, 38]}, {grid[0, 17], grid[0, 19]}, {grid[0, 3], grid[0, 46]},
                   {grid[0, 0], grid[0, 44]},
                   {grid[0, 21], grid[0, 48]}, {grid[0, 10], grid[0, 41]}, {grid[0, 45], grid[0, 47]},
                   {grid[0, 16], grid[0, 23]},
                   {grid[0, 8], grid[0, 18]}, {grid[0, 42], grid[0, 44]}, {grid[0, 2], grid[0, 20]},
                   {grid[0, 14], grid[0, 19]},
                   {grid[0, 2], grid[0, 23]}, {grid[0, 8], grid[0, 19]}, {grid[0, 19], grid[0, 33]},
                   {grid[0, 8], grid[0, 11]},
                   {grid[0, 9], grid[0, 24]}, {grid[0, 0], grid[0, 26]}, {grid[0, 17], grid[0, 29]},
                   {grid[0, 10], grid[0, 30]},
                   {grid[0, 11], grid[0, 49]}, {grid[0, 30], grid[0, 42]}, {grid[0, 10], grid[0, 22]},
                   {grid[0, 29], grid[0, 34]},
                   {grid[0, 25], grid[0, 36]}, {grid[0, 24], grid[0, 42]}, {grid[0, 5], grid[0, 48]},
                   {grid[0, 29], grid[0, 45]},
                   {grid[0, 5], grid[0, 26]}, {grid[0, 14], grid[0, 33]}, {grid[0, 35], grid[0, 46]},
                   {grid[0, 4], grid[0, 13]},
                   {grid[0, 18], grid[0, 45]}, {grid[0, 8], grid[0, 25]}, {grid[0, 3], grid[0, 40]},
                   {grid[0, 26], grid[0, 44]},
                   {grid[0, 34], grid[0, 39]}, {grid[0, 16], grid[0, 34]}, {grid[0, 5], grid[0, 20]},
                   {grid[0, 2], grid[0, 37]},
                   {grid[0, 14], grid[0, 25]}, {grid[0, 27], grid[0, 38]}, {grid[0, 4], grid[0, 26]},
                   {grid[0, 17], grid[0, 22]},
                   {grid[0, 27], grid[0, 28]}, {grid[0, 4], grid[0, 47]}, {grid[0, 9], grid[0, 10]},
                   {grid[0, 25], grid[0, 39]},
                   {grid[0, 30], grid[0, 48]}, {grid[0, 30], grid[0, 39]}, {grid[0, 12], grid[0, 47]},
                   {grid[0, 28], grid[0, 48]},
                   {grid[0, 41], grid[0, 44]}, {grid[0, 48], grid[0, 49]}, {grid[0, 18], grid[0, 41]},
                   {grid[0, 3], grid[0, 4]},
                   {grid[0, 7], grid[0, 45]}, {grid[0, 29], grid[0, 38]}]

    for i, j in constraints:
        model += i != j

    C_T = list(model.constraints)

    print(len(C_T))

    return grid, C_T, model


def construct_random495():
    # Variables
    grid = intvar(1, 5, shape=(1, 100), name="grid")

    model = Model()

    # 495 constraints randomly generated
    scopes = [{grid[0, 68], grid[0, 97]}, {grid[0, 67], grid[0, 99]}, {grid[0, 84], grid[0, 94]},
              {grid[0, 15], grid[0, 92]}, {grid[0, 2], grid[0, 40]}, {grid[0, 27], grid[0, 39]},
              {grid[0, 53], grid[0, 61]}, {grid[0, 29], grid[0, 85]}, {grid[0, 14], grid[0, 33]},
              {grid[0, 31], grid[0, 36]}, {grid[0, 22], grid[0, 69]}, {grid[0, 21], grid[0, 25]},
              {grid[0, 36], grid[0, 76]}, {grid[0, 1], grid[0, 98]}, {grid[0, 52], grid[0, 91]},
              {grid[0, 44], grid[0, 97]}, {grid[0, 43], grid[0, 48]}, {grid[0, 12], grid[0, 32]},
              {grid[0, 1], grid[0, 13]}, {grid[0, 85], grid[0, 96]}, {grid[0, 15], grid[0, 40]},
              {grid[0, 27], grid[0, 85]}, {grid[0, 45], grid[0, 51]}, {grid[0, 58], grid[0, 90]},
              {grid[0, 23], grid[0, 57]}, {grid[0, 23], grid[0, 78]}, {grid[0, 88], grid[0, 94]},
              {grid[0, 73], grid[0, 81]}, {grid[0, 9], grid[0, 34]}, {grid[0, 60], grid[0, 87]},
              {grid[0, 64], grid[0, 76]}, {grid[0, 14], grid[0, 63]}, {grid[0, 15], grid[0, 84]},
              {grid[0, 2], grid[0, 37]}, {grid[0, 7], grid[0, 52]}, {grid[0, 11], grid[0, 69]},
              {grid[0, 0], grid[0, 71]}, {grid[0, 21], grid[0, 27]}, {grid[0, 38], grid[0, 89]},
              {grid[0, 40], grid[0, 91]}, {grid[0, 12], grid[0, 99]}, {grid[0, 85], grid[0, 87]},
              {grid[0, 5], grid[0, 29]}, {grid[0, 3], grid[0, 64]}, {grid[0, 42], grid[0, 94]},
              {grid[0, 21], grid[0, 34]}, {grid[0, 37], grid[0, 57]}, {grid[0, 59], grid[0, 81]},
              {grid[0, 58], grid[0, 77]}, {grid[0, 24], grid[0, 66]}, {grid[0, 2], grid[0, 17]},
              {grid[0, 3], grid[0, 20]}, {grid[0, 76], grid[0, 96]}, {grid[0, 54], grid[0, 85]},
              {grid[0, 51], grid[0, 68]}, {grid[0, 8], grid[0, 94]}, {grid[0, 10], grid[0, 61]},
              {grid[0, 2], grid[0, 21]}, {grid[0, 24], grid[0, 42]}, {grid[0, 8], grid[0, 48]},
              {grid[0, 45], grid[0, 94]}, {grid[0, 7], grid[0, 48]}, {grid[0, 37], grid[0, 42]},
              {grid[0, 34], grid[0, 72]}, {grid[0, 20], grid[0, 36]}, {grid[0, 97], grid[0, 98]},
              {grid[0, 42], grid[0, 55]}, {grid[0, 91], grid[0, 99]}, {grid[0, 9], grid[0, 31]},
              {grid[0, 28], grid[0, 95]}, {grid[0, 4], grid[0, 45]}, {grid[0, 22], grid[0, 88]},
              {grid[0, 15], grid[0, 25]}, {grid[0, 17], grid[0, 22]}, {grid[0, 49], grid[0, 51]},
              {grid[0, 26], grid[0, 35]}, {grid[0, 26], grid[0, 42]}, {grid[0, 72], grid[0, 96]},
              {grid[0, 42], grid[0, 91]}, {grid[0, 72], grid[0, 81]}, {grid[0, 36], grid[0, 85]},
              {grid[0, 60], grid[0, 91]}, {grid[0, 28], grid[0, 52]}, {grid[0, 40], grid[0, 70]},
              {grid[0, 57], grid[0, 75]}, {grid[0, 27], grid[0, 87]}, {grid[0, 73], grid[0, 75]},
              {grid[0, 73], grid[0, 95]}, {grid[0, 16], grid[0, 70]}, {grid[0, 94], grid[0, 95]},
              {grid[0, 46], grid[0, 80]}, {grid[0, 73], grid[0, 94]}, {grid[0, 30], grid[0, 91]},
              {grid[0, 25], grid[0, 53]}, {grid[0, 24], grid[0, 75]}, {grid[0, 30], grid[0, 56]},
              {grid[0, 63], grid[0, 64]}, {grid[0, 53], grid[0, 56]}, {grid[0, 44], grid[0, 49]},
              {grid[0, 85], grid[0, 90]}, {grid[0, 36], grid[0, 73]}, {grid[0, 63], grid[0, 95]},
              {grid[0, 9], grid[0, 47]}, {grid[0, 2], grid[0, 5]}, {grid[0, 75], grid[0, 91]},
              {grid[0, 72], grid[0, 82]}, {grid[0, 8], grid[0, 42]}, {grid[0, 3], grid[0, 75]},
              {grid[0, 11], grid[0, 79]}, {grid[0, 25], grid[0, 26]}, {grid[0, 66], grid[0, 74]},
              {grid[0, 14], grid[0, 90]}, {grid[0, 16], grid[0, 26]}, {grid[0, 26], grid[0, 84]},
              {grid[0, 41], grid[0, 84]}, {grid[0, 18], grid[0, 32]}, {grid[0, 7], grid[0, 82]},
              {grid[0, 0], grid[0, 35]}, {grid[0, 3], grid[0, 60]}, {grid[0, 27], grid[0, 90]},
              {grid[0, 64], grid[0, 78]}, {grid[0, 50], grid[0, 93]}, {grid[0, 65], grid[0, 74]},
              {grid[0, 66], grid[0, 99]}, {grid[0, 50], grid[0, 68]}, {grid[0, 34], grid[0, 76]},
              {grid[0, 2], grid[0, 46]}, {grid[0, 6], grid[0, 44]}, {grid[0, 34], grid[0, 98]},
              {grid[0, 24], grid[0, 30]}, {grid[0, 15], grid[0, 51]}, {grid[0, 22], grid[0, 44]},
              {grid[0, 58], grid[0, 93]}, {grid[0, 66], grid[0, 77]}, {grid[0, 57], grid[0, 92]},
              {grid[0, 2], grid[0, 74]}, {grid[0, 36], grid[0, 62]}, {grid[0, 49], grid[0, 89]},
              {grid[0, 26], grid[0, 96]}, {grid[0, 36], grid[0, 64]}, {grid[0, 5], grid[0, 7]},
              {grid[0, 55], grid[0, 87]}, {grid[0, 60], grid[0, 76]}, {grid[0, 14], grid[0, 66]},
              {grid[0, 64], grid[0, 94]}, {grid[0, 25], grid[0, 51]}, {grid[0, 60], grid[0, 70]},
              {grid[0, 16], grid[0, 34]}, {grid[0, 29], grid[0, 94]}, {grid[0, 2], grid[0, 56]},
              {grid[0, 67], grid[0, 89]}, {grid[0, 17], grid[0, 89]}, {grid[0, 32], grid[0, 38]},
              {grid[0, 88], grid[0, 89]}, {grid[0, 29], grid[0, 48]}, {grid[0, 6], grid[0, 40]},
              {grid[0, 92], grid[0, 96]}, {grid[0, 45], grid[0, 74]}, {grid[0, 20], grid[0, 89]},
              {grid[0, 27], grid[0, 72]}, {grid[0, 18], grid[0, 62]}, {grid[0, 85], grid[0, 94]},
              {grid[0, 23], grid[0, 64]}, {grid[0, 39], grid[0, 49]}, {grid[0, 14], grid[0, 24]},
              {grid[0, 50], grid[0, 56]}, {grid[0, 13], grid[0, 38]}, {grid[0, 15], grid[0, 86]},
              {grid[0, 61], grid[0, 88]}, {grid[0, 28], grid[0, 79]}, {grid[0, 31], grid[0, 62]},
              {grid[0, 33], grid[0, 68]}, {grid[0, 5], grid[0, 85]}, {grid[0, 38], grid[0, 39]},
              {grid[0, 6], grid[0, 75]}, {grid[0, 1], grid[0, 33]}, {grid[0, 0], grid[0, 13]},
              {grid[0, 45], grid[0, 53]}, {grid[0, 48], grid[0, 94]}, {grid[0, 20], grid[0, 93]},
              {grid[0, 57], grid[0, 68]}, {grid[0, 49], grid[0, 75]}, {grid[0, 38], grid[0, 93]},
              {grid[0, 34], grid[0, 54]}, {grid[0, 72], grid[0, 89]}, {grid[0, 34], grid[0, 61]},
              {grid[0, 70], grid[0, 88]}, {grid[0, 78], grid[0, 82]}, {grid[0, 81], grid[0, 84]},
              {grid[0, 39], grid[0, 76]}, {grid[0, 17], grid[0, 50]}, {grid[0, 16], grid[0, 58]},
              {grid[0, 24], grid[0, 96]}, {grid[0, 28], grid[0, 44]}, {grid[0, 74], grid[0, 83]},
              {grid[0, 75], grid[0, 83]}, {grid[0, 18], grid[0, 72]}, {grid[0, 6], grid[0, 45]},
              {grid[0, 69], grid[0, 89]}, {grid[0, 1], grid[0, 73]}, {grid[0, 21], grid[0, 87]},
              {grid[0, 39], grid[0, 73]}, {grid[0, 65], grid[0, 66]}, {grid[0, 8], grid[0, 78]},
              {grid[0, 12], grid[0, 78]}, {grid[0, 48], grid[0, 64]}, {grid[0, 11], grid[0, 73]},
              {grid[0, 7], grid[0, 74]}, {grid[0, 43], grid[0, 75]}, {grid[0, 1], grid[0, 54]},
              {grid[0, 10], grid[0, 83]}, {grid[0, 22], grid[0, 99]}, {grid[0, 15], grid[0, 98]},
              {grid[0, 33], grid[0, 94]}, {grid[0, 41], grid[0, 71]}, {grid[0, 47], grid[0, 81]},
              {grid[0, 22], grid[0, 86]}, {grid[0, 18], grid[0, 27]}, {grid[0, 19], grid[0, 30]},
              {grid[0, 6], grid[0, 70]}, {grid[0, 54], grid[0, 77]}, {grid[0, 31], grid[0, 96]},
              {grid[0, 43], grid[0, 46]}, {grid[0, 48], grid[0, 68]}, {grid[0, 96], grid[0, 99]},
              {grid[0, 78], grid[0, 99]}, {grid[0, 93], grid[0, 98]}, {grid[0, 39], grid[0, 89]},
              {grid[0, 5], grid[0, 49]}, {grid[0, 2], grid[0, 95]}, {grid[0, 37], grid[0, 68]},
              {grid[0, 34], grid[0, 35]}, {grid[0, 1], grid[0, 15]}, {grid[0, 13], grid[0, 23]},
              {grid[0, 63], grid[0, 77]}, {grid[0, 62], grid[0, 82]}, {grid[0, 2], grid[0, 19]},
              {grid[0, 4], grid[0, 69]}, {grid[0, 30], grid[0, 41]}, {grid[0, 28], grid[0, 39]},
              {grid[0, 24], grid[0, 46]}, {grid[0, 1], grid[0, 25]}, {grid[0, 74], grid[0, 89]},
              {grid[0, 17], grid[0, 84]}, {grid[0, 0], grid[0, 65]}, {grid[0, 35], grid[0, 84]},
              {grid[0, 66], grid[0, 80]}, {grid[0, 14], grid[0, 88]}, {grid[0, 8], grid[0, 93]},
              {grid[0, 6], grid[0, 47]}, {grid[0, 42], grid[0, 64]}, {grid[0, 0], grid[0, 80]},
              {grid[0, 76], grid[0, 92]}, {grid[0, 25], grid[0, 33]}, {grid[0, 73], grid[0, 80]},
              {grid[0, 69], grid[0, 98]}, {grid[0, 17], grid[0, 74]}, {grid[0, 36], grid[0, 72]},
              {grid[0, 9], grid[0, 41]}, {grid[0, 33], grid[0, 82]}, {grid[0, 25], grid[0, 43]},
              {grid[0, 45], grid[0, 71]}, {grid[0, 17], grid[0, 48]}, {grid[0, 42], grid[0, 92]},
              {grid[0, 8], grid[0, 15]}, {grid[0, 11], grid[0, 91]}, {grid[0, 36], grid[0, 53]},
              {grid[0, 34], grid[0, 43]}, {grid[0, 44], grid[0, 68]}, {grid[0, 64], grid[0, 96]},
              {grid[0, 0], grid[0, 57]}, {grid[0, 25], grid[0, 28]}, {grid[0, 9], grid[0, 49]},
              {grid[0, 23], grid[0, 36]}, {grid[0, 1], grid[0, 68]}, {grid[0, 12], grid[0, 50]},
              {grid[0, 51], grid[0, 84]}, {grid[0, 0], grid[0, 91]}, {grid[0, 7], grid[0, 80]},
              {grid[0, 10], grid[0, 90]}, {grid[0, 11], grid[0, 53]}, {grid[0, 3], grid[0, 52]},
              {grid[0, 19], grid[0, 75]}, {grid[0, 27], grid[0, 56]}, {grid[0, 4], grid[0, 51]},
              {grid[0, 72], grid[0, 90]}, {grid[0, 40], grid[0, 82]}, {grid[0, 25], grid[0, 75]},
              {grid[0, 64], grid[0, 71]}, {grid[0, 8], grid[0, 80]}, {grid[0, 46], grid[0, 63]},
              {grid[0, 19], grid[0, 81]}, {grid[0, 80], grid[0, 84]}, {grid[0, 47], grid[0, 50]},
              {grid[0, 41], grid[0, 62]}, {grid[0, 61], grid[0, 93]}, {grid[0, 47], grid[0, 54]},
              {grid[0, 60], grid[0, 83]}, {grid[0, 78], grid[0, 93]}, {grid[0, 95], grid[0, 96]},
              {grid[0, 20], grid[0, 71]}, {grid[0, 48], grid[0, 82]}, {grid[0, 3], grid[0, 45]},
              {grid[0, 83], grid[0, 95]}, {grid[0, 10], grid[0, 22]}, {grid[0, 38], grid[0, 40]},
              {grid[0, 31], grid[0, 50]}, {grid[0, 32], grid[0, 82]}, {grid[0, 56], grid[0, 90]},
              {grid[0, 40], grid[0, 64]}, {grid[0, 46], grid[0, 95]}, {grid[0, 1], grid[0, 83]},
              {grid[0, 2], grid[0, 43]}, {grid[0, 18], grid[0, 28]}, {grid[0, 31], grid[0, 60]},
              {grid[0, 43], grid[0, 79]}, {grid[0, 17], grid[0, 68]}, {grid[0, 19], grid[0, 93]},
              {grid[0, 36], grid[0, 43]}, {grid[0, 13], grid[0, 67]}, {grid[0, 98], grid[0, 99]},
              {grid[0, 15], grid[0, 37]}, {grid[0, 0], grid[0, 25]}, {grid[0, 45], grid[0, 47]},
              {grid[0, 40], grid[0, 94]}, {grid[0, 61], grid[0, 97]}, {grid[0, 0], grid[0, 97]},
              {grid[0, 40], grid[0, 66]}, {grid[0, 90], grid[0, 94]}, {grid[0, 67], grid[0, 69]},
              {grid[0, 5], grid[0, 96]}, {grid[0, 5], grid[0, 17]}, {grid[0, 19], grid[0, 97]},
              {grid[0, 25], grid[0, 85]}, {grid[0, 19], grid[0, 41]}, {grid[0, 23], grid[0, 76]},
              {grid[0, 76], grid[0, 98]}, {grid[0, 50], grid[0, 69]}, {grid[0, 0], grid[0, 67]},
              {grid[0, 5], grid[0, 34]}, {grid[0, 42], grid[0, 76]}, {grid[0, 21], grid[0, 37]},
              {grid[0, 3], grid[0, 18]}, {grid[0, 25], grid[0, 56]}, {grid[0, 20], grid[0, 82]},
              {grid[0, 65], grid[0, 94]}, {grid[0, 40], grid[0, 45]}, {grid[0, 0], grid[0, 23]},
              {grid[0, 69], grid[0, 85]}, {grid[0, 31], grid[0, 49]}, {grid[0, 76], grid[0, 78]},
              {grid[0, 29], grid[0, 98]}, {grid[0, 31], grid[0, 72]}, {grid[0, 22], grid[0, 68]},
              {grid[0, 55], grid[0, 69]}, {grid[0, 14], grid[0, 38]}, {grid[0, 12], grid[0, 22]},
              {grid[0, 28], grid[0, 71]}, {grid[0, 57], grid[0, 58]}, {grid[0, 35], grid[0, 82]},
              {grid[0, 12], grid[0, 83]}, {grid[0, 17], grid[0, 34]}, {grid[0, 41], grid[0, 51]},
              {grid[0, 4], grid[0, 91]}, {grid[0, 75], grid[0, 84]}, {grid[0, 1], grid[0, 87]},
              {grid[0, 23], grid[0, 77]}, {grid[0, 69], grid[0, 71]}, {grid[0, 25], grid[0, 65]},
              {grid[0, 44], grid[0, 58]}, {grid[0, 16], grid[0, 59]}, {grid[0, 54], grid[0, 82]},
              {grid[0, 0], grid[0, 4]}, {grid[0, 31], grid[0, 80]}, {grid[0, 28], grid[0, 74]},
              {grid[0, 62], grid[0, 90]}, {grid[0, 77], grid[0, 84]}, {grid[0, 24], grid[0, 29]},
              {grid[0, 10], grid[0, 88]}, {grid[0, 34], grid[0, 44]}, {grid[0, 52], grid[0, 73]},
              {grid[0, 47], grid[0, 62]}, {grid[0, 1], grid[0, 91]}, {grid[0, 27], grid[0, 38]},
              {grid[0, 57], grid[0, 85]}, {grid[0, 58], grid[0, 73]}, {grid[0, 55], grid[0, 97]},
              {grid[0, 71], grid[0, 95]}, {grid[0, 49], grid[0, 50]}, {grid[0, 52], grid[0, 85]},
              {grid[0, 16], grid[0, 32]}, {grid[0, 17], grid[0, 20]}, {grid[0, 67], grid[0, 79]},
              {grid[0, 37], grid[0, 81]}, {grid[0, 27], grid[0, 76]}, {grid[0, 61], grid[0, 79]},
              {grid[0, 42], grid[0, 71]}, {grid[0, 7], grid[0, 69]}, {grid[0, 53], grid[0, 84]},
              {grid[0, 17], grid[0, 31]}, {grid[0, 24], grid[0, 56]}, {grid[0, 43], grid[0, 66]},
              {grid[0, 72], grid[0, 87]}, {grid[0, 10], grid[0, 30]}, {grid[0, 30], grid[0, 64]},
              {grid[0, 60], grid[0, 78]}, {grid[0, 36], grid[0, 52]}, {grid[0, 12], grid[0, 23]},
              {grid[0, 23], grid[0, 66]}, {grid[0, 16], grid[0, 53]}, {grid[0, 24], grid[0, 25]},
              {grid[0, 58], grid[0, 87]}, {grid[0, 41], grid[0, 79]}, {grid[0, 19], grid[0, 52]},
              {grid[0, 14], grid[0, 73]}, {grid[0, 16], grid[0, 68]}, {grid[0, 9], grid[0, 63]},
              {grid[0, 12], grid[0, 38]}, {grid[0, 51], grid[0, 85]}, {grid[0, 35], grid[0, 70]},
              {grid[0, 36], grid[0, 87]}, {grid[0, 27], grid[0, 84]}, {grid[0, 18], grid[0, 23]},
              {grid[0, 14], grid[0, 49]}, {grid[0, 5], grid[0, 47]}, {grid[0, 19], grid[0, 32]},
              {grid[0, 5], grid[0, 16]}, {grid[0, 30], grid[0, 39]}, {grid[0, 56], grid[0, 71]},
              {grid[0, 40], grid[0, 59]}, {grid[0, 6], grid[0, 32]}, {grid[0, 69], grid[0, 97]},
              {grid[0, 38], grid[0, 43]}, {grid[0, 9], grid[0, 22]}, {grid[0, 46], grid[0, 89]},
              {grid[0, 54], grid[0, 92]}, {grid[0, 37], grid[0, 71]}, {grid[0, 39], grid[0, 74]},
              {grid[0, 68], grid[0, 86]}, {grid[0, 37], grid[0, 89]}, {grid[0, 82], grid[0, 98]},
              {grid[0, 51], grid[0, 76]}, {grid[0, 60], grid[0, 62]}, {grid[0, 19], grid[0, 73]},
              {grid[0, 52], grid[0, 84]}, {grid[0, 44], grid[0, 95]}, {grid[0, 39], grid[0, 91]},
              {grid[0, 1], grid[0, 81]}, {grid[0, 15], grid[0, 97]}, {grid[0, 9], grid[0, 38]},
              {grid[0, 29], grid[0, 36]}, {grid[0, 41], grid[0, 52]}, {grid[0, 59], grid[0, 69]},
              {grid[0, 68], grid[0, 90]}, {grid[0, 30], grid[0, 42]}, {grid[0, 6], grid[0, 79]},
              {grid[0, 21], grid[0, 65]}, {grid[0, 45], grid[0, 59]}, {grid[0, 17], grid[0, 33]},
              {grid[0, 8], grid[0, 69]}, {grid[0, 40], grid[0, 96]}, {grid[0, 55], grid[0, 73]},
              {grid[0, 31], grid[0, 99]}, {grid[0, 18], grid[0, 35]}, {grid[0, 45], grid[0, 55]},
              {grid[0, 76], grid[0, 95]}, {grid[0, 58], grid[0, 86]}, {grid[0, 42], grid[0, 90]},
              {grid[0, 10], grid[0, 34]}, {grid[0, 8], grid[0, 38]}, {grid[0, 22], grid[0, 74]},
              {grid[0, 11], grid[0, 75]}, {grid[0, 24], grid[0, 69]}, {grid[0, 24], grid[0, 53]},
              {grid[0, 2], grid[0, 53]}, {grid[0, 18], grid[0, 98]}, {grid[0, 26], grid[0, 83]},
              {grid[0, 10], grid[0, 69]}, {grid[0, 9], grid[0, 40]}, {grid[0, 64], grid[0, 85]},
              {grid[0, 13], grid[0, 52]}, {grid[0, 57], grid[0, 81]}, {grid[0, 16], grid[0, 23]},
              {grid[0, 8], grid[0, 59]}, {grid[0, 83], grid[0, 99]}, {grid[0, 17], grid[0, 95]},
              {grid[0, 54], grid[0, 56]}, {grid[0, 16], grid[0, 79]}, {grid[0, 1], grid[0, 89]},
              {grid[0, 46], grid[0, 58]}, {grid[0, 15], grid[0, 89]}, {grid[0, 49], grid[0, 70]},
              {grid[0, 49], grid[0, 91]}, {grid[0, 70], grid[0, 77]}, {grid[0, 1], grid[0, 86]}, ]

    for i, j in scopes:
        model += i != j

    C_T = list(model.constraints)

    print(len(C_T))

    return grid, C_T, model


def construct_golomb8():
    # Variables
    grid = intvar(1, 35, shape=(1, 8), name="grid")

    model = Model()

    for i in range(8):
        for j in range(i + 1, 8):
            for x in range(j + 1, 7):
                for y in range(x + 1, 8):
                    if y != i and x != j and x != i and y != j:
                        model += abs(grid[0, i] - grid[0, j]) != abs(grid[0, x] - grid[0, y])

    C_T = list(model.constraints)

    print(len(C_T))

    return grid, C_T, model


def construct_murder_problem():
    # Variables
    grid = intvar(1, 5, shape=(4, 5), name="grid")

    C_T = list()

    # Constraints on rows and columns
    model = Model([AllDifferent(row).decompose() for row in grid])

    # Additional constraints of the murder problem
    C_T += [grid[0, 1] == grid[1, 2]]
    C_T += [grid[0, 2] != grid[1, 4]]
    C_T += [grid[3, 2] != grid[1, 4]]
    C_T += [grid[0, 2] != grid[1, 0]]
    C_T += [grid[0, 2] != grid[3, 4]]
    C_T += [grid[3, 4] == grid[1, 3]]
    C_T += [grid[1, 1] == grid[2, 1]]
    C_T += [grid[2, 3] == grid[0, 3]]
    C_T += [grid[2, 0] == grid[3, 3]]
    C_T += [grid[0, 0] != grid[2, 4]]
    C_T += [grid[0, 0] != grid[1, 4]]
    C_T += [grid[0, 0] == grid[3, 0]]

    model += C_T

    for row in grid:
        C_T += list(AllDifferent(row).decompose())

    C_T = toplevel_list(C_T)

    return grid, C_T, model


def construct_job_shop_scheduling_problem(n_jobs, machines, horizon, seed=0):
    random.seed(seed)
    max_time = horizon // n_jobs

    duration = [[0] * machines for i in range(0, n_jobs)]
    for i in range(0, n_jobs):
        for j in range(0, machines):
            duration[i][j] = random.randint(1, max_time)

    task_to_mach = [list(range(0, machines)) for i in range(0, n_jobs)]

    for i in range(0, n_jobs):
        random.shuffle(task_to_mach[i])

    precedence = [[(i, j) for j in task_to_mach[i]] for i in range(0, n_jobs)]

    # convert to numpy
    task_to_mach = np.array(task_to_mach)
    duration = np.array(duration)
    precedence = np.array(precedence)

    machines = set(task_to_mach.flatten().tolist())

    model = cp.Model()

    # decision variables
    start = cp.intvar(1, horizon, shape=task_to_mach.shape, name="start")
    end = cp.intvar(1, horizon, shape=task_to_mach.shape, name="end")

    grid = cp.cpm_array(np.expand_dims(np.concatenate([start.flatten(), end.flatten()]), 0))

    # precedence constraints
    for chain in precedence:
        for (j1, t1), (j2, t2) in zip(chain[:-1], chain[1:]):
            model += end[j1, t1] <= start[j2, t2]

    # duration constraints
    model += (start + duration == end)

    # non_overlap constraints per machine
    for m in machines:
        tasks_on_mach = np.where(task_to_mach == m)
        for (j1, t1), (j2, t2) in all_pairs(zip(*tasks_on_mach)):
            m += (end[j1, t1] <= start[j2, t2]) | (end[j2, t2] <= start[j1, t1])

    C = list(model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    temp = []
    for c in C:
        if isinstance(c, cp.expressions.core.Comparison):
            temp.append(c)
        elif isinstance(c, cp.expressions.variables.NDVarArray):
            _c = c.flatten()
            for __c in _c:
                temp.append(__c)
    # [temp.append(c) for c in C]
    C_T = set(temp)

    max_duration = max(duration)
    return grid, C_T, model, max_duration
