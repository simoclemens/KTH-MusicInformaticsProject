class NNMixing:
    def __init__(self, track_list, number_of_tracks=5, key_weight=0.25):
        self.track_list = track_list
        self.number_of_tracks = number_of_tracks
        self.key_weight = key_weight

    # compute the distance in terms of BPM
    def computeDistance(self, track1, track2):
        bpm_delta = abs(track1.bpm-track2.bpm)
        key_delta = self.getCamelotDist(track1.key_idx, track2.key_idx)
        return bpm_delta + self.key_weight * key_delta

    # fond the next track to be put in the lineup considering key and BPM
    def nnSearch(self, track_idx):
        current = self.track_list[track_idx]

        min_dist = float("+inf")
        min_idx = None

        for i, elem in enumerate(self.track_list):
            if not elem.selected and current.bpm < elem.bpm:
                dist = self.computeDistance(current, elem)
                if dist < min_dist:
                    min_idx = i
                    min_dist = dist

        return min_idx

    # iteratively creates the lineup fo the mix
    def createLineup(self):
        lineup = []
        min_idx = -1
        min_bpm = float("+inf")
        n_songs = len(self.track_list)

        for i, track in enumerate(self.track_list):
            if track.bpm < min_bpm:
                min_idx = i
                min_bpm = track.bpm

        current_idx = min_idx

        for i in range(self.number_of_tracks):
            lineup.append(self.track_list[current_idx])
            self.track_list[current_idx].setSelected()
            current_idx = self.nnSearch(current_idx)

        return lineup

    def getCamelotDist(self, current_key, next_key):
        current_minor = False
        next_minor = False

        minor_circle = [16, 8, 20, 10, 0, 14, 4, 18, 8, 22, 12, 2]
        major_circle = [23, 13, 3, 17, 7, 21, 11, 1, 15, 5, 19, 9]

        if current_key in minor_circle:
            start_idx = minor_circle.index(current_key)
            current_minor = True
        else:
            start_idx = major_circle.index(current_key)

        if next_key in minor_circle:
            arrive_idx = minor_circle.index(next_key)
            next_minor = True
        else:
            arrive_idx = major_circle.index(next_key)

        abs_dist = abs(start_idx - arrive_idx)
        circ_dist = min(abs_dist, 12 - abs_dist)

        if current_minor != next_minor:
            circ_dist += 1

        return circ_dist

    def getCamelotKeys(self, current):
        key = current.key_idx
        output = [key]
        minor_circle = [16, 8, 20, 10, 0, 14, 4, 18, 8, 22, 12, 2]
        major_circle = [23, 13, 3, 17, 7, 21, 11, 1, 15, 5, 19, 9]
        allowed_tracks=[]
        i=1

        while len(allowed_tracks) == 0:

            if key in minor_circle:
                index = minor_circle.index(key)

                next_idx = index + i
                if next_idx > 11: next_idx = i-11
                prev_idx = index - i
                if prev_idx < 0: next_idx = 12-i
                output.append(major_circle[index])
                output.append(minor_circle[prev_idx])
                output.append(minor_circle[next_idx])

            else:
                index = major_circle.index(key)

                next_idx = index + 1
                if next_idx > 11: next_idx = 0
                prev_idx = index - 1
                if prev_idx < 0: next_idx = 11
                output.append(minor_circle[index])
                output.append(major_circle[prev_idx])
                output.append(major_circle[next_idx])

            # for track in self.track_list:


        return output






