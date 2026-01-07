package main

import "github.com/jonasknobloch/mbpe"

func surveyResponses(name string) ([]float64, []bool, []string, error) {
	var data map[string]interface{}

	if err := mbpe.FromJSON(name, &data); err != nil {
		return nil, nil, nil, err
	}

	ratios := make([]float64, 200)
	binary := make([]bool, 200)

	keys := getKeys(data)

	for i, key := range keys {
		responsesData := data[key].(map[string]interface{})

		ity := 0
		ness := 0

		for _, v := range responsesData {
			s, ok := v.(string)

			if !ok {
				continue
			}

			if s == "ity" {
				ity += 1

				continue
			}

			ness += 1
		}

		ratios[i] = float64(ity) / float64(ity+ness) // 1.0 for all "ity"
		binary[i] = ratios[i] > 0.5                  // true for "ity"; false for "ness"
	}

	return ratios, binary, keys, nil
}
