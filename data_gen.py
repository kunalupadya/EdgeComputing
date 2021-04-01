import numpy as np
# import matplotlib.pyplot as plt
import datetime
import bidict
import pickle

# locations_list = [
locations_dict = {
    "Class West": 0,
    "Class East": 1,
    "Class Pratt": 2,
    "Residence East": 3,
    "Residence West": 4,
    "Residence Off West": 5,
    "Residence Off East": 6,
    "Residence Downtown": 7,
    "Bus stop West": 8,
    "Bus stop East": 9,
    "West Union": 10,
    "BC Plaza": 11,
    "BC": 12,
    "Equad": 13,
    "Chapel": 14,
    "Under WU": 15,
    "Perkins": 16,
    "Marketplace": 17,
    "Lilly": 18,
    "GADU": 19,
    "Wilson": 20,
    "Taishoff": 21,
    "Cameron": 22,
    "KVille": 23,
    "Brodie": 24,
    "Brodie Aquatic": 25,
    "Brodie Tennis": 26,
    "Wilson Tennis": 27,
    "Pitchforks": 28
}

# locations_dict = {}
# for l in locations_list:
#     locations_dict[l] = np.random.random()

locations = bidict.bidict(locations_dict)


def norm_pdf(x, mu=0, sigma=1):
    """
    Probability density function (pdf) for the normal distribution
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html

    Params:
        x (np.ndarray): random variable
        mu (float): mean
        sigma (float): standard deviation

    Returns:
        norm_pdf (np.ndarray): the result of the pdf
    """
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))


class Student():
    """docstring for Student"""

    def __init__(self, id=0):
        self.id = id
        self.randomize_attributes()
        self.randomize_schedule()

    def __repr__(self):
        return f"Student {self.id}:\nSchedule: {self.schedule}"

    def randomize_attributes(self):
        # choose which grade (freshman = 0, senior = 3)
        # Even split of grades is approximately correct
        self.grade = np.random.randint(4)

        # from that, choose on campus or not
        # Percent chances pulled out of thin air
        n = np.random.random()
        if self.grade == 0:
            self.on_campus = n < 0.9
        elif self.grade == 1:
            self.on_campus = n < 0.6
        elif self.grade == 2:
            self.on_campus = n < 0.3
        elif self.grade == 3:
            self.on_campus = n < 0.1

        # from that, choose where they're living
        # 0 = "East", 1 = "West", 2 = "off West", 3 = "off East", 4 = "downtown"
        # This obviously is an oversimplification, and could be improved
        n = np.random.random()
        if self.on_campus:
            if self.grade == 0:
                self.residence = "Residence East" if n < 0.7 else "Residence West"
            else:
                # RAs might live on East
                self.residence = "Residence East" if n < 0.05 else "Residence West"
        else:
            if self.grade == 0:
                if n < 0.05:
                    self.residence = "Residence Off West"
                elif n < 0.9:
                    self.residence = "Residence Off East"
                else:
                    self.residence = "Residence Downtown"
            elif self.grade == 1:
                if n < 0.4:
                    self.residence = "Residence Off West"
                elif n < 0.8:
                    self.residence = "Residence Off East"
                else:
                    self.residence = "Residence Downtown"
            elif self.grade == 2:
                if n < 0.3:
                    self.residence = "Residence Off West"
                elif n < 0.8:
                    self.residence = "Residence Off East"
                else:
                    self.residence = "Residence Downtown"
            elif self.grade == 3:
                if n < 0.4:
                    self.residence = "Residence Off West"
                elif n < 0.9:
                    self.residence = "Residence Off East"
                else:
                    self.residence = "Residence Downtown"

        # from living situation, choose % of meals from campus eateries
        if self.on_campus:
            self.campus_meals = np.clip(np.random.normal(loc=0.75, scale=0.1), 0, 1)
        else:
            if self.residence == "Residence Off West":
                self.campus_meals = np.clip(np.random.normal(loc=0.35, scale=0.1), 0, 1)
            elif self.residence == "Residence Off East":
                self.campus_meals = np.clip(np.random.normal(loc=0.3, scale=0.1), 0, 1)
            elif self.residence == "Residence Downtown":
                self.campus_meals = np.clip(np.random.normal(loc=0.25, scale=0.1), 0, 1)

        # num in-person classes
        # self.in_person_classes = np.floor(np.clip(np.random.normal(loc=1.5, scale=1), 0, 4))

        # hours of sleep
        self.hours_of_sleep = np.floor(np.clip(np.random.normal(loc=9, scale=0.7), 4, 12))

        # class attendance %
        self.class_attendance = np.clip(np.random.normal(loc=0.8, scale=0.2), 0, 1)

        # social score
        self.social = np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)

        # studious score
        self.studious = np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)

        # athletic score
        self.athletic = np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)

        # major?

    def randomize_schedule(self):
        # start date: Monday, Jan 25, 2021
        self.start_date = datetime.datetime(2021, 1, 25)
        # end date: Friday, Apr 23, 2021

        self.schedule = []

        self.generate_classes()
        self.generate_sleep()

        self.sort_schedule()

        self.generate_meals()

    def sort_schedule(self):
        self.schedule.sort(key=lambda event: event.start)

    def generate_classes(self):
        # number of classes
        num_classes = 0
        n = np.random.random()
        if n < 0.1:
            num_classes = 2
        elif n < 0.15:
            num_classes = 3
        elif n < 0.3:
            num_classes = 5
        else:
            num_classes = 4

        # attributes
        fraction_online = 0.7
        fraction_of_online_asynchronous = 0.3
        fraction_lab = 0.2
        fraction_discussion = 0.2

        # location if in-person
        fraction_pratt = 0.1
        fraction_east = 0.15
        fraction_west = 0.75

        # classes start on the quarter hour, last 1.25 (3 for lab)
        # assume 2 lectures a week (M-W, T-Th, W-F), maybe discussion and/or lab
        lecture_length = 1.5  # hours
        discussion_length = 1.5  # hours
        lab_length = 3  # hours
        lecture_day_options = ['M-W', 'T-Th', 'W-F']
        earliest_class_start = 8  # 0800
        latest_class_start = 17  # 1700

        # actually create classes
        i = 0
        while i < num_classes:
            online = np.random.random() < fraction_online
            asynchronous = online and np.random.random() < fraction_of_online_asynchronous
            has_lab = np.random.random() < fraction_lab
            has_discussion = np.random.random() < fraction_discussion

            lecture_days = lecture_day_options[np.random.randint(3)]

            location_id = locations[self.residence]
            if not online:
                n = np.random.random()
                if n < fraction_pratt:
                    location_id = locations["Class Pratt"]
                elif n < fraction_pratt + fraction_east:
                    location_id = locations["Class East"]
                else:
                    location_id = locations["Class West"]

            # create events
            if not asynchronous:
                lecture1_start = None
                lecture_start_hour = np.random.randint(earliest_class_start, latest_class_start)
                lecture_start_minute = 15 * np.random.randint(3)
                if lecture_days == 'M-W':
                    lecture1_start = datetime.datetime(2021, 1, 25, lecture_start_hour, lecture_start_minute)
                elif lecture_days == 'T-Th':
                    lecture1_start = datetime.datetime(2021, 1, 26, lecture_start_hour, lecture_start_minute)
                elif lecture_days == 'W-F':
                    lecture1_start = datetime.datetime(2021, 1, 27, lecture_start_hour, lecture_start_minute)

                lecture2_start = lecture1_start + datetime.timedelta(days=2)

                lecture1_end = lecture1_start + datetime.timedelta(hours=lecture_length)
                lecture2_end = lecture2_start + datetime.timedelta(hours=lecture_length)

                lecture1 = Event(f"lecture{i}a", lecture1_start, lecture1_end, location_id)
                lecture2 = Event(f"lecture{i}b", lecture2_start, lecture2_end, location_id)

            if has_lab:
                lab = None
                while not lab or not asynchronous and (lab.conflicts_with(lecture1) or lab.conflicts_with(lecture2)):
                    start_hour = np.random.randint(earliest_class_start, latest_class_start)
                    start_minute = 15 * np.random.randint(3)
                    start = datetime.datetime(2021, 1, 25, start_hour, start_minute)
                    start += datetime.timedelta(days=np.random.randint(4))
                    end = start + datetime.timedelta(hours=lab_length)
                    lab = Event(f"lab{i}", start, end, location_id)

            if has_discussion:
                discussion = None
                while not discussion \
                        or not asynchronous \
                        and (discussion.conflicts_with(lecture1) or discussion.conflicts_with(lecture2)) \
                        or has_lab and discussion.conflicts_with(lab):
                    start_hour = np.random.randint(earliest_class_start, latest_class_start)
                    start_minute = 15 * np.random.randint(3)
                    start = datetime.datetime(2021, 1, 25, start_hour, start_minute)
                    start += datetime.timedelta(days=np.random.randint(4))
                    end = start + datetime.timedelta(hours=discussion_length)
                    discussion = Event(f"discussion{i}", start, end, location_id)

            conflict = False
            for e in self.schedule:
                if not asynchronous and (e.conflicts_with(lecture1) or e.conflicts_with(lecture2)) \
                        or has_lab and e.conflicts_with(lab) \
                        or has_discussion and e.conflicts_with(discussion):
                    conflict = True
                    break

            if not conflict:
                if not asynchronous:
                    self.schedule.append(lecture1)
                    self.schedule.append(lecture2)
                if has_lab:
                    self.schedule.append(lab)
                if has_discussion:
                    self.schedule.append(discussion)
                i += 1

    def generate_sleep(self):
        earliest_sleep_start = 21  # 2100
        latest_sleep_start = 23  # 2300

        for i in range(25, 30):
            sleep_start_hour = np.random.randint(earliest_sleep_start, latest_sleep_start)
            sleep_start_minute = 15 * np.random.randint(3)
            sleep_start = datetime.datetime(2021, 1, i, sleep_start_hour, sleep_start_minute)
            sleep_end = sleep_start + datetime.timedelta(hours=self.hours_of_sleep)
            location_id = locations[self.residence]
            sleep = Event(f"sleep{i}", sleep_start, sleep_end, location_id)
            self.schedule.append(sleep)

    def generate_meals(self):
        """
        Assuming a schedule sorted chronologically so far, generate meals.
        Shooting for longest meals possible for simplicity.
        """
        shortest_meal = 0.5
        longest_meal = 2

        shortest_meal_td = datetime.timedelta(hours=shortest_meal)
        longest_meal_td = datetime.timedelta(hours=longest_meal)
        half_longest_meal_td = datetime.timedelta(hours=longest_meal / 2)

        meal_mid_goals = [("breakfast", 8), ("lunch", 12), ("dinner", 19)]

        for i in range(25, 30):
            for name, j in meal_mid_goals:
                goal_mid = datetime.datetime(2021, 1, i, j)
                window = self.find_window(goal_mid)
                if not window:  # no time, skip meal
                    continue
                wstart, wend, first, second = window
                diff = wend - wstart

                # TODO: Need to actually find location for this meal
                m_location_id = locations["West Union"]

                if diff < shortest_meal_td:  # too short, skip meal
                    continue

                if diff <= longest_meal_td:  # just right
                    meal = Event(name, wstart, wend, m_location_id)
                elif wend - goal_mid < half_longest_meal_td:  # weighted earlier
                    mstart = wend - longest_meal_td
                    meal = Event(name, mstart, wend, m_location_id)
                elif goal_mid - wstart < half_longest_meal_td:  # weighted later
                    mend = wstart + longest_meal_td
                    meal = Event(name, wstart, mend, m_location_id)
                else:  # tons of time
                    mstart = goal_mid - half_longest_meal_td
                    mend = goal_mid + half_longest_meal_td
                    meal = Event(name, mstart, mend, m_location_id)

                self.schedule.insert(second, meal)

    def find_window(self, dt):
        """Tries to find largest window of unscheduled time including datetime dt
        Returns (start, end) of window, or None if not possible
        Assumes schedule is sorted chronologically.
        """
        for i, e in enumerate(self.schedule):
            if e.end < dt:
                continue
            if e.start < dt and dt < e.end:
                return None

            # at this point, e.start > dt, so figure out and return
            if i == 0:
                start = self.start_date
            else:
                start = self.schedule[i - 1].end
            return start, e.start, i - 1, i

    def get_locations_each_hour(self):
        """
        Returns location ids and the hours [0, 23] at which they take place.
        Days are ignored, i.e. hour 2 on a given day is the same as hour 2 on
        the next day.
        Weekend hours are [24, 47].
        Assumes self.schedule is chronologically.

        Example:
        location_ids = [45, 32, 10, 0, 2, 3, 7, 32]
        hours = [1, 10, 13, 23, 1, 2, 9, 14]

        Returns: tuple of (location_ids, hours)
        """
        location_ids = []
        hours = []
        for i, e in enumerate(self.schedule):
            dt = e.start - datetime.timedelta(minutes=e.start.minute,
                                              seconds=e.start.second,
                                              microseconds=e.start.microsecond)
            while dt < e.end:
                location_ids.append(e.location_id)
                hours.append(dt.hour if dt.weekday() <= 4 else dt.hour + 24)  # Account for weekends
                dt += datetime.timedelta(hours=1)
        return location_ids, hours


class Event():
    """docstring for Event"""

    def __init__(self, name, start, end, location_id, id=None):
        """start and end are datetime objects"""
        self.name = name
        self.id = id or np.random.random()
        self.start = start
        self.end = end
        self.location_id = location_id

    def __repr__(self):
        return f"Event {self.id}: name: {self.name}, start: {self.start}, end: {self.end}, location: {locations.inverse[self.location_id]}"

    def conflicts_with(self, event):
        return self.start < event.end and self.end > event.start


def main():
    # Create students
    students = []
    num_students = 20
    for i in range(num_students):
        students.append(Student(id=i))

    # Create student sessions with a sliding window
    # session = (loc, tim, tar)
    # loc is list of locations, tim is list of times, tar is loc offset one forward
    student_sessions = {}  # {student_id: [session0, session1]}
    sess_len = 15  # 15 location records per session
    for s in students:
        location_ids, hours = s.get_locations_each_hour()
        sessions = []
        for i in range(len(location_ids) - sess_len):
            sessions.append((location_ids[i:i+sess_len], hours[i:i+sess_len], location_ids[i+1:i+sess_len+1]))
        student_sessions[s.id] = sessions

    # print stuff
    print(students[0].get_locations_each_hour())
    print(student_sessions)

    # pickle stuff
    with open("data.pickle", "wb") as f:
        pickle.dump(student_sessions, f)

    # n = np.sort(np.random.normal(size=1000))
    # plt.plot(n)
    # # plt.plot(n, norm_pdf(n))
    # plt.show()
    # print(np.mean(n), np.std(n, ddof=1))


if __name__ == '__main__':
    main()
