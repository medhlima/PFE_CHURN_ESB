// ==========================================================
// CHURNAI - MAIN FRONTEND LOGIC
// ==========================================================


// ==========================================================
// UTILITY
// ==========================================================

function formatFeatureName(feature) {

    return feature
        .replaceAll("_", " ")
        .toLowerCase()
        .replace(
            /\b\w/g,
            letter => letter.toUpperCase()
        );
}


// ==========================================================
// ELEMENTS
// ==========================================================

const predictionForm =
    document.getElementById("predictionForm");

const loadCustomerButton =
    document.getElementById("loadCustomer");

const customerSearch =
    document.getElementById("customerSearch");


// ==========================================================
// CUSTOMER SEARCH + AUTO-FILL
// ==========================================================

if (
    loadCustomerButton &&
    customerSearch &&
    predictionForm
) {

    loadCustomerButton.addEventListener(
        "click",
        async function () {

            const customerId =
                customerSearch.value.trim();

            if (!customerId) {

                alert(
                    "Please enter a Customer ID."
                );

                return;
            }

            try {

                loadCustomerButton.disabled = true;

                loadCustomerButton.textContent =
                    "Loading...";

                const response = await fetch(
                    `/customer/${encodeURIComponent(customerId)}`
                );

                const customer =
                    await response.json();

                if (!response.ok) {

                    throw new Error(
                        customer.error ||
                        "Customer could not be loaded."
                    );
                }

                const booleanFields = [
                    "HAS_SALAIRE",
                    "EST_TUNISIEN",
                    "EST_RESIDENT",
                    "DOSSIER_COMPLET",
                    "KYC_RISQUE_ELEVE",
                    "REVUE_EN_RETARD",
                    "A_HISTORIQUE_REVUE"
                ];

                Object.entries(
                    customer
                ).forEach(
                    ([key, value]) => {

                        if (
                            key === "CUSTOMER_NO"
                        ) {
                            return;
                        }

                        const field =
                            predictionForm
                                .elements[key];

                        if (!field) {
                            return;
                        }

                        if (
                            booleanFields.includes(
                                key
                            )
                        ) {

                            field.checked =
                                Boolean(value);

                        } else {

                            field.value =
                                value ?? "";

                        }
                    }
                );

                alert(
                    `Customer ${customer.CUSTOMER_NO} loaded successfully.`
                );

            }

            catch (error) {

                alert(
                    error.message
                );

            }

            finally {

                loadCustomerButton.disabled =
                    false;

                loadCustomerButton.textContent =
                    "Load Customer";
            }
        }
    );
}


// ==========================================================
// CUSTOMER CHURN PREDICTION
// ==========================================================

if (predictionForm) {

    predictionForm.addEventListener(
        "submit",
        async function (event) {

            event.preventDefault();

            const formData =
                new FormData(
                    predictionForm
                );

            const payload = {};


            // --------------------------------------------------
            // NUMERICAL FEATURES
            // --------------------------------------------------

            const numericFields = [
                "AGE",
                "ANCIENNETE_CLIENT_ANNEES",
                "SALAIRE",
                "JOURS_AVANT_PROCHAINE_REVUE",
                "NB_COMPTES"
            ];


            // --------------------------------------------------
            // BOOLEAN FEATURES
            // --------------------------------------------------

            const booleanFields = [
                "HAS_SALAIRE",
                "EST_TUNISIEN",
                "EST_RESIDENT",
                "DOSSIER_COMPLET",
                "KYC_RISQUE_ELEVE",
                "REVUE_EN_RETARD",
                "A_HISTORIQUE_REVUE"
            ];


            // --------------------------------------------------
            // BUILD PAYLOAD
            // --------------------------------------------------

            for (
                const [key, value]
                of formData.entries()
            ) {

                if (
                    numericFields.includes(
                        key
                    )
                ) {

                    payload[key] =
                        value === ""
                            ? null
                            : Number(value);

                } else {

                    payload[key] =
                        value;
                }
            }


            booleanFields.forEach(
                field => {

                    payload[field] =
                        predictionForm
                            .elements[field]
                            .checked;
                }
            );


            // --------------------------------------------------
            // API REQUEST
            // --------------------------------------------------

            try {

                const response =
                    await fetch(
                        "/predict",
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body:
                                JSON.stringify(
                                    payload
                                )
                        }
                    );

                const result =
                    await response.json();

                if (!response.ok) {

                    throw new Error(
                        result.error ||
                        "Prediction failed."
                    );
                }


                // --------------------------------------------------
                // SHOW RESULT PANEL
                // --------------------------------------------------

                const emptyResult =
                    document.getElementById(
                        "emptyResult"
                    );

                const resultContent =
                    document.getElementById(
                        "resultContent"
                    );

                if (emptyResult) {

                    emptyResult
                        .classList
                        .add("hidden");
                }

                if (resultContent) {

                    resultContent
                        .classList
                        .remove("hidden");
                }


                // --------------------------------------------------
                // PROBABILITY
                // --------------------------------------------------

                const probabilityValue =
                    document.getElementById(
                        "probabilityValue"
                    );

                if (probabilityValue) {

                    probabilityValue.textContent =
                        result
                            .churn_probability_percent
                        + "%";
                }


                // --------------------------------------------------
                // PREDICTION LABEL
                // --------------------------------------------------

                const predictionLabel =
                    document.getElementById(
                        "predictionLabel"
                    );

                if (predictionLabel) {

                    predictionLabel.textContent =
                        result.prediction_label;
                }


                // --------------------------------------------------
                // RISK LEVEL
                // --------------------------------------------------

                const riskElement =
                    document.getElementById(
                        "riskLevel"
                    );

                if (riskElement) {

                    riskElement.textContent =
                        result.risk_level;

                    riskElement.className = "";

                    if (
                        result.risk_level ===
                        "HIGH"
                    ) {

                        riskElement
                            .classList
                            .add(
                                "risk-high"
                            );

                    }

                    else if (
                        result.risk_level ===
                        "MEDIUM"
                    ) {

                        riskElement
                            .classList
                            .add(
                                "risk-medium"
                            );

                    }

                    else {

                        riskElement
                            .classList
                            .add(
                                "risk-low"
                            );
                    }
                }


                // ==================================================
                // SHAP EXPLAINABILITY
                // ==================================================

                const explanationContainer =
                    document.getElementById(
                        "explanationFactors"
                    );

                if (
                    explanationContainer
                ) {

                    explanationContainer
                        .innerHTML = "";

                    const explanations =
                        result.explanation || [];

                    if (
                        explanations.length > 0
                    ) {

                        const maxImpact =
                            Math.max(
                                ...explanations.map(
                                    item =>
                                        item.impact
                                )
                            );

                        explanations.forEach(
                            item => {

                                const percentage =
                                    maxImpact > 0
                                        ? (
                                            item.impact
                                            /
                                            maxImpact
                                        ) * 100
                                        : 0;

                                const increasesRisk =
                                    item.direction
                                    ===
                                    "INCREASES_RISK";

                                const factor =
                                    document.createElement(
                                        "div"
                                    );

                                factor.className =
                                    "explanation-factor";

                                factor.innerHTML = `
                                    <div class="factor-top">

                                        <span class="factor-name">
                                            ${
                                                formatFeatureName(
                                                    item.feature
                                                )
                                            }
                                        </span>

                                        <span class="
                                            factor-direction
                                            ${
                                                increasesRisk
                                                    ? "factor-increase"
                                                    : "factor-decrease"
                                            }
                                        ">
                                            ${
                                                increasesRisk
                                                    ? "↑ Increases risk"
                                                    : "↓ Reduces risk"
                                            }
                                        </span>

                                    </div>

                                    <div class="factor-bar">

                                        <div
                                            class="
                                                factor-bar-fill
                                                ${
                                                    increasesRisk
                                                        ? "factor-bar-increase"
                                                        : "factor-bar-decrease"
                                                }
                                            "
                                            style="
                                                width:
                                                ${percentage}%
                                            "
                                        ></div>

                                    </div>
                                `;

                                explanationContainer
                                    .appendChild(
                                        factor
                                    );
                            }
                        );

                    } else {

                        explanationContainer.innerHTML =
                            `
                            <p style="
                                color:
                                var(--text-secondary);
                                font-size: 12px;
                            ">
                                No explanation available.
                            </p>
                            `;
                    }
                }

            }

            catch (error) {

                alert(
                    error.message
                );
            }
        }
    );
}
const customer360Search =
    document.getElementById("customer360Search");

const loadCustomer360 =
    document.getElementById("loadCustomer360");

const customer360Content =
    document.getElementById("customer360Content");


function yesNo(value) {
    return value ? "Yes" : "No";
}


function displayValue(value) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "Not available";
    }

    return value;
}


if (
    customer360Search &&
    loadCustomer360 &&
    customer360Content
) {

    loadCustomer360.addEventListener(
        "click",
        async function () {

            const customerId =
                customer360Search.value.trim();

            if (!customerId) {
                alert("Please enter a Customer ID.");
                return;
            }

            try {

                loadCustomer360.disabled = true;
                loadCustomer360.textContent =
                    "Loading...";

                const response = await fetch(
                    `/customer/${encodeURIComponent(customerId)}`
                );

                const customer =
                    await response.json();

                if (!response.ok) {
                    throw new Error(
                        customer.error ||
                        "Customer not found."
                    );
                }

                document.getElementById(
                    "c360CustomerId"
                ).textContent =
                    customer.CUSTOMER_NO;

                document.getElementById(
                    "c360Type"
                ).textContent =
                    displayValue(
                        customer.TYPE_CLIENT
                    );

                document.getElementById(
                    "c360Segment"
                ).textContent =
                    displayValue(
                        customer.SEGMENT
                    );

                document.getElementById(
                    "c360Age"
                ).textContent =
                    displayValue(
                        customer.AGE
                    );

                document.getElementById(
                    "c360AgeGroup"
                ).textContent =
                    displayValue(
                        customer.TRANCHE_AGE
                    );

                document.getElementById(
                    "c360Tenure"
                ).textContent =
                    customer.ANCIENNETE_CLIENT_ANNEES !== null
                        ? `${customer.ANCIENNETE_CLIENT_ANNEES} years`
                        : "Not available";

                document.getElementById(
                    "c360Accounts"
                ).textContent =
                    displayValue(
                        customer.NB_COMPTES
                    );

                document.getElementById(
                    "c360HasSalary"
                ).textContent =
                    yesNo(
                        customer.HAS_SALAIRE
                    );

                document.getElementById(
                    "c360Salary"
                ).textContent =
                    displayValue(
                        customer.SALAIRE
                    );

                document.getElementById(
                    "c360Kyc"
                ).textContent =
                    displayValue(
                        customer.SCORE_KYC
                    );

                document.getElementById(
                    "c360KycRisk"
                ).textContent =
                    yesNo(
                        customer.KYC_RISQUE_ELEVE
                    );

                document.getElementById(
                    "c360CompleteFile"
                ).textContent =
                    yesNo(
                        customer.DOSSIER_COMPLET
                    );

                document.getElementById(
                    "c360ReviewOverdue"
                ).textContent =
                    yesNo(
                        customer.REVUE_EN_RETARD
                    );

                document.getElementById(
                    "c360NextReview"
                ).textContent =
                    formatReviewStatus(
                        customer.JOURS_AVANT_PROCHAINE_REVUE
                    );

                document.getElementById(
                    "c360Tunisian"
                ).textContent =
                    yesNo(
                        customer.EST_TUNISIEN
                    );

                document.getElementById(
                    "c360Resident"
                ).textContent =
                    yesNo(
                        customer.EST_RESIDENT
                    );

                document.getElementById(
                    "c360Marital"
                ).textContent =
                    formatMaritalStatus(
                        customer.SITUATION_FAMILIALE
                    );

                document.getElementById(
                    "c360ReviewHistory"
                ).textContent =
                    yesNo(
                        customer.A_HISTORIQUE_REVUE
                    );

                // ==========================================================
                // CUSTOMER 360 - CHURN RISK SUMMARY
                // ==========================================================

                const predictionResponse = await fetch(
                    "/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type": "application/json"
                        },

                        body: JSON.stringify(customer)
                    }
                );

                const predictionResult =
                    await predictionResponse.json();

                if (!predictionResponse.ok) {

                    throw new Error(
                        predictionResult.error ||
                        "Unable to calculate churn risk."
                    );
                }


                // ----------------------------------------------------------
                // CHURN PROBABILITY
                // ----------------------------------------------------------

                const c360Probability =
                    document.getElementById(
                        "c360Probability"
                    );

                if (c360Probability) {

                    c360Probability.textContent =
                        predictionResult
                            .churn_probability_percent
                        + "%";
                }


                // ----------------------------------------------------------
                // PREDICTION
                // ----------------------------------------------------------

                const c360Prediction =
                    document.getElementById(
                        "c360Prediction"
                    );

                if (c360Prediction) {

                    c360Prediction.textContent =
                        predictionResult.prediction_label;
                }


                // ----------------------------------------------------------
                // RISK LEVEL
                // ----------------------------------------------------------

                const c360RiskLevel =
                    document.getElementById(
                        "c360RiskLevel"
                    );

                if (c360RiskLevel) {

                    c360RiskLevel.textContent =
                        predictionResult.risk_level;

                    c360RiskLevel.className = "";

                    if (
                        predictionResult.risk_level === "HIGH"
                    ) {

                        c360RiskLevel.classList.add(
                            "risk-high"
                        );

                    } else if (
                        predictionResult.risk_level === "MEDIUM"
                    ) {

                        c360RiskLevel.classList.add(
                            "risk-medium"
                        );

                    } else {

                        c360RiskLevel.classList.add(
                            "risk-low"
                        );
                    }
                }     
                customer360Content
                    .classList
                    .remove("hidden");

            }

            catch (error) {

                alert(error.message);

            }

            finally {

                loadCustomer360.disabled = false;
                loadCustomer360.textContent =
                    "Load Profile";
            }
        }
    );
    
}
function formatReviewStatus(days) {

    if (
        days === null ||
        days === undefined ||
        days === ""
    ) {
        return "Not available";
    }

    const value = Number(days);

    if (value > 0) {
        return `Review due in ${value} days`;
    }

    if (value === 0) {
        return "Review due today";
    }

    return `Overdue by ${Math.abs(value)} days`;
}


function formatMaritalStatus(value) {

    const mapping = {
        "M": "Married",
        "C": "Single",
        "D": "Divorced",
        "S": "Separated"
    };

    return mapping[value] || value || "Not available";
}
// ==========================================================
// PORTFOLIO ANALYTICS
// ==========================================================

const analyticsContent =
    document.getElementById(
        "analyticsContent"
    );

const analyticsLoading =
    document.getElementById(
        "analyticsLoading"
    );


function renderAnalyticsBars(
    containerId,
    rows,
    labelKey
) {

    const container =
        document.getElementById(
            containerId
        );

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (!rows || rows.length === 0) {

        container.innerHTML =
            "<p>No data available.</p>";

        return;
    }

    const maxRate = Math.max(
        ...rows.map(
            row => row.churn_rate
        )
    );

    rows.forEach(row => {

        const relativeWidth =
            maxRate > 0
                ? (
                    row.churn_rate
                    / maxRate
                ) * 100
                : 0;

        const item =
            document.createElement(
                "div"
            );

        item.className =
            "analytics-bar-item";

        item.innerHTML = `
            <div class="analytics-bar-header">

                <span class="analytics-bar-label">
                    ${row[labelKey]}
                </span>

                <span class="analytics-bar-value">
                    ${row.churn_rate.toFixed(2)}%
                </span>

            </div>

            <div class="analytics-bar-track">

                <div
                    class="analytics-bar-fill"
                    style="width:${relativeWidth}%"
                ></div>

            </div>

            <div class="analytics-bar-meta">

                <span>
                    ${row.total_customers.toLocaleString()}
                    customers
                </span>

                <span>
                    ${row.churn_customers.toLocaleString()}
                    churners
                </span>

            </div>
        `;

        container.appendChild(
            item
        );
    });
}


async function loadPortfolioAnalytics() {

    if (
        !analyticsContent ||
        !analyticsLoading
    ) {
        return;
    }

    try {

        const response = await fetch(
            "/api/analytics"
        );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.error ||
                "Unable to load analytics."
            );
        }


        // ------------------------------
        // KPIs
        // ------------------------------

        document.getElementById(
            "analyticsTotalCustomers"
        ).textContent =
            data.portfolio
                .total_customers
                .toLocaleString();


        document.getElementById(
            "analyticsChurnCustomers"
        ).textContent =
            data.portfolio
                .churn_customers
                .toLocaleString();


        document.getElementById(
            "analyticsNonChurnCustomers"
        ).textContent =
            data.portfolio
                .non_churn_customers
                .toLocaleString();


        document.getElementById(
            "analyticsChurnRate"
        ).textContent =
            data.portfolio
                .churn_rate
                .toFixed(2)
            + "%";


        // ------------------------------
        // SEGMENTATIONS
        // ------------------------------

        renderAnalyticsBars(
            "segmentAnalytics",
            data.by_segment,
            "SEGMENT"
        );

        renderAnalyticsBars(
            "typeAnalytics",
            data.by_customer_type,
            "TYPE_CLIENT"
        );

        renderAnalyticsBars(
            "ageAnalytics",
            data.by_age_group,
            "TRANCHE_AGE"
        );

        renderAnalyticsBars(
            "kycAnalytics",
            data.by_kyc,
            "SCORE_KYC"
        );


        analyticsLoading
            .classList
            .add("hidden");

        analyticsContent
            .classList
            .remove("hidden");

    }

    catch (error) {

        analyticsLoading.textContent =
            error.message;
    }
}


loadPortfolioAnalytics();
// ==========================================================
// MODEL GOVERNANCE
// ==========================================================

const governanceContent =
    document.getElementById(
        "governanceContent"
    );

const governanceLoading =
    document.getElementById(
        "governanceLoading"
    );


async function loadModelGovernance() {

    if (
        !governanceContent ||
        !governanceLoading
    ) {
        return;
    }

    try {

        const response = await fetch(
            "/api/governance"
        );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.error ||
                "Unable to load governance metadata."
            );
        }


        // ==================================================
        // MODEL
        // ==================================================

        document.getElementById(
            "govModelName"
        ).textContent =
            data.model.registered_name;


        document.getElementById(
            "govRegisteredModel"
        ).textContent =
            data.model.registered_name;


        document.getElementById(
            "govAlgorithm"
        ).textContent =
            data.model.algorithm;


        document.getElementById(
            "govRegistryVersion"
        ).textContent =
            data.model.version
                ? `Version ${data.model.version}`
                : "Not registered";


        document.getElementById(
            "govVersion"
        ).textContent =
            data.model.version
                ? `Version ${data.model.version}`
                : "No version";


        document.getElementById(
            "govRegistryState"
        ).textContent =
            data.model.registry_status;


        document.getElementById(
            "govRegistryStatus"
        ).textContent =
            data.model.registry_status;


        // ==================================================
        // EXPERIMENT / RUN
        // ==================================================

        document.getElementById(
            "govExperiment"
        ).textContent =
            data.experiment.name;


        document.getElementById(
            "govRunName"
        ).textContent =
            data.run.name;


        document.getElementById(
            "govRunStatus"
        ).textContent =
            data.run.status;


        document.getElementById(
            "govRunDate"
        ).textContent =
            data.run.created_at ||
            "Not available";


        // ==================================================
        // METRICS
        // ==================================================

        document.getElementById(
            "govRocAuc"
        ).textContent =
            data.metrics.roc_auc.toFixed(4);


        document.getElementById(
            "govPrAuc"
        ).textContent =
            data.metrics.pr_auc.toFixed(4);


        document.getElementById(
            "govRecall"
        ).textContent =
            data.metrics.recall.toFixed(4);


        document.getElementById(
            "govPrecision"
        ).textContent =
            data.metrics.precision.toFixed(4);


        document.getElementById(
            "govF1"
        ).textContent =
            data.metrics.f1.toFixed(4);


        document.getElementById(
            "govAccuracy"
        ).textContent =
            data.metrics.accuracy.toFixed(4);


        // ==================================================
        // DEPLOYMENT
        // ==================================================

        document.getElementById(
            "govPredictionService"
        ).textContent =
            data.deployment
                .prediction_service;


        document.getElementById(
            "govThreshold"
        ).textContent =
            (
                data.deployment
                    .decision_threshold
                * 100
            ).toFixed(0)
            + "%";


        document.getElementById(
            "govExplainability"
        ).textContent =
            data.deployment
                .explainability;


        document.getElementById(
            "govApiStatus"
        ).textContent =
            data.deployment
                .api_status;


        governanceLoading
            .classList
            .add("hidden");


        governanceContent
            .classList
            .remove("hidden");

    }

    catch (error) {

        governanceLoading.textContent =
            error.message;
    }
}


loadModelGovernance();
// ==========================================================
// EXECUTIVE DASHBOARD
// ==========================================================

async function loadExecutiveDashboard() {

    const totalCustomers =
        document.getElementById("dashboardTotalCustomers");

    // If we are not on the dashboard page, stop here.
    if (!totalCustomers) {
        return;
    }

    try {

        // --------------------------------------------------
        // LOAD ANALYTICS DATA
        // --------------------------------------------------

        const analyticsResponse =
            await fetch("/api/analytics");

        const analyticsData =
            await analyticsResponse.json();

        if (!analyticsResponse.ok) {
            throw new Error(
                analyticsData.error ||
                "Unable to load analytics data."
            );
        }


        // --------------------------------------------------
        // LOAD GOVERNANCE / MODEL DATA
        // --------------------------------------------------

        const governanceResponse =
            await fetch("/api/governance");

        const governanceData =
            await governanceResponse.json();

        if (!governanceResponse.ok) {
            throw new Error(
                governanceData.error ||
                "Unable to load model data."
            );
        }


        // ==================================================
        // PORTFOLIO KPIs
        // ==================================================

        document.getElementById(
            "dashboardTotalCustomers"
        ).textContent =
            analyticsData.portfolio
                .total_customers
                .toLocaleString();


        document.getElementById(
            "dashboardChurnRate"
        ).textContent =
            analyticsData.portfolio
                .churn_rate
                .toFixed(2)
            + "%";


        document.getElementById(
            "dashboardChurnDetail"
            
        ).textContent =
            analyticsData.portfolio
                .churn_customers
                .toLocaleString()
            + " churn customers";


        // ==================================================
        // MODEL KPIs
        // ==================================================

        document.getElementById(
            "dashboardRocAuc"
        ).textContent =
            governanceData.metrics
                .roc_auc
                .toFixed(4);


        document.getElementById(
            "dashboardPrAuc"
        ).textContent =
            governanceData.metrics
                .pr_auc
                .toFixed(4);


        document.getElementById(
            "dashboardRecall"
        ).textContent =
            governanceData.metrics
                .recall
                .toFixed(4);


        document.getElementById(
            "dashboardPrecision"
        ).textContent =
            governanceData.metrics
                .precision
                .toFixed(4);


        document.getElementById(
            "dashboardF1"
        ).textContent =
            governanceData.metrics
                .f1
                .toFixed(4);


        document.getElementById(
            "dashboardModel"
        ).textContent =
            governanceData.model.algorithm;

    }

    catch (error) {

        console.error(
            "Dashboard loading error:",
            error
        );
    }
}


loadExecutiveDashboard();