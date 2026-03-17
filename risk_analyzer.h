#pragma once

#include "yolo_trt.h"

#include <string>
#include <vector>

struct PersonRiskResult {
    Detection person_box;
    bool has_helmet = false;
    bool has_vest = false;
    bool is_risk = false;
    std::string warning_message;
};

struct SceneRiskSummary {
    int worker_count = 0;
    int no_helmet_count = 0;
    int no_vest_count = 0;
    bool has_any_risk = false;
};

class RiskAnalyzer {
public:
    explicit RiskAnalyzer(float contain_ratio_thresh = 0.35f);

    std::vector<PersonRiskResult> analyzePPE(const std::vector<Detection>& dets) const;
    bool hasAnyRisk(const std::vector<PersonRiskResult>& results) const;
    std::string resultsToText(const std::vector<PersonRiskResult>& results) const;

    SceneRiskSummary summarizeScene(const std::vector<PersonRiskResult>& results) const;
    std::string sceneSummaryToText(const SceneRiskSummary& summary) const;

private:
    float contain_ratio_thresh_;

    float intersectionArea(const Detection& a, const Detection& b) const;
    float boxArea(const Detection& b) const;
    float centerX(const Detection& b) const;
    float centerY(const Detection& b) const;
    bool pointInBox(float x, float y, const Detection& box) const;

    Detection makeHelmetRegion(const Detection& person) const;
    Detection makeVestRegion(const Detection& person) const;

    float calcOverlapRatio(const Detection& region, const Detection& item) const;
    float calcCenterDxRatio(const Detection& person, const Detection& item) const;

    bool isHelmetMatch(
        const Detection& person,
        const Detection& region,
        const Detection& helmet
    ) const;

    bool isVestMatch(
        const Detection& person,
        const Detection& region,
        const Detection& vest
    ) const;
};