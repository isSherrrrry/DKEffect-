var graph;

var defaultX = 0, defaultY = 1;
var x0 = null, x0old = null, x1 = null, dims = [];
var attrNo = null, attr = null, attr2 = [], index = 0;
var X = [], Y = [];
var loaddata = [];
var istxtdata = false;
var colorScale = d3.scale.category10();
var label = { DEFAULT: "Un-Assign", PG: "Point Guard", SG: "Shooting Guard", SF: "Small Forward", PF: "Power Forward", C: "Center" };
var positions = [label.DEFAULT, label.PG, label.SG, label.SF, label.PF, label.C];
var colorMap = { "Un-Assign": "#7f7f7f", "Point Guard": colorScale(0), "Shooting Guard": colorScale(1), "Small Forward": colorScale(2), "Power Forward": colorScale(3), "Center": colorScale(4) };
var defaultColor = colorScale(5);
defaultColor = colorScale(6);
defaultColor = colorScale(7);
var positionDescriptions1 = { "Un-Assign": "un-assign a point", "Point Guard": "usually the smallest and quickest players", "Shooting Guard": "typically of small-medium size and stature", "Small Forward": "typically of medium size and stature", "Power Forward": "typically of medium-large size and stature", "Center": "typically the largest players on the team" };
var positionDescriptions2 = { "Un-Assign": "un-assign a point", "Point Guard": "skilled at passing and dribbling; primarily responsible for distributing the ball to other players resulting in many assists", "Shooting Guard": "typically attempts many shots, especially long-ranged shots", "Small Forward": "typically a strong defender with lots of steals", "Power Forward": "typically spends most time near the basket, resulting in lots of rebounds", "Center": "responsible for protecting the basket, resulting in lots of blocks" };
//var positionDescriptionsDemo = {"Un-Assign": "un-assign a point", "Point Guard": "responsible for controlling the ball", "Shooting Guard": "guards the opponent's best perimeter player on defense", "Small Forward": "typically makes many rebounds", "Power Forward": "typically some of the physically strongest players on the team", "Center": "typically relied on for both strong offense and defense"};
var positionDescriptions;// = positionDescriptions1;
var activePosition = "none";
var playerPositionMap = {};
var helpMouseoverStart = 0;
var helpMouseoverEnd = 0;

//Gets called when the page is loaded.
function init() {
	// defaultX = $("#initX :selected").text();
	// defaultY = $("#initY :selected").text();
	$("#area1").show();
	$("#area2").show();
	$("#area3").show();
	$("#dialog").hide();
	loadData();
}
function loadData() {
	// Get input data
	d3.csv('data/bball_top50_decimal_removed.csv', function (data) { // demo50 dataset
		var loaddata = jQuery.extend(true, [], data);
		for (var i = 0; i < loaddata.length; i++) {
			loaddata[i]["Name"] = "Player " + loaddata[i]["Player Anonymized"];
			delete loaddata[i]["Player"];
			delete loaddata[i]["Player Anonymized"];
			delete loaddata[i]["Position"];
			delete loaddata[i]["Team"];
			playerPositionMap[loaddata[i]["Name"]] = "none";
			loaddata[i]["coord"] = {};
		}
		attr = Object.keys(loaddata[0]);
		attr.pop(); // remove "coord" from attribute list
		attr.pop(); // remove "Name" from attribute list
		attrNo = attr.length;

		// set default user labels
		loaddata.forEach(function (d) {
			d["coord"]["userlabel"] = label.DEFAULT;
		});

		for (var i = 0; i < attrNo; i++) {
			if (attr[i] != "Name" && attr[i] != "coord") {
				var tmpmax = d3.max(loaddata, function (d) { return +d[attr[i]]; });
				var tmpmin = d3.min(loaddata, function (d) { return +d[attr[i]]; });

				// jitter the value by adding a random number up to +/- 3% of scale
				function jitter(val, attribute) {
					var mult = 0.03;
					if (attribute == "Height (Inches)") mult = 0.02;
					if (attribute == "Weight (Pounds)") mult = 0.01;
					var noise = Math.random() * mult * (tmpmax - tmpmin);

					// determine whether to add or subtract
					var sign = Math.round(Math.random());
					if (sign == 0) return val - noise;
					else return val + noise;
				}

				loaddata.forEach(function (d) {
					d["coord"][attr[i]] = jitter(+d[attr[i]], attr[i]);// jitter((+d[attr[i]]-tmpmin)/(tmpmax-tmpmin), attr[i]);
				});
			}
		}

		// determine which condition the user follows and which set of descriptions should be used
		var rand = Math.floor(Math.random() * 10) + 1;
		console.log(rand);
		if (rand % 2 == 0) positionDescriptions = positionDescriptions1;
		else positionDescriptions = positionDescriptions2;
		// console.log(window.localStorage.getItem("whichCondition"));
		// if (window.localStorage.getItem("whichCondition") == 1) positionDescriptions = positionDescriptions1;
		// else positionDescriptions = positionDescriptions2;

		// initialize log entries
		R7Insight.init({
			token: '8c524878-4ee8-43c5-91f5-ffa797c3726a',
			region: 'us'
		});
		R7Insight.log("Log event");

		// initialize IAL
		ial.init(loaddata, 0, ["coord", "Name"], "exclude", -1, 1);
		console.log("ial initialized");

		// load the vis
		loadVis(loaddata);
	});
}

//Main function
function loadVis(data) {
	drawScatterPlot(data);
	//updateBias(true);

	for (var i = 0; i < attrNo; i++) {
		dims[i] = attr[i];
	}
	drawParaCoords(data, dims);
	tabulate(data[0], 'empty');
	addHelp();
	addClassificationControls(data);
	addCustomAxisDropDownControls();
}

function addCustomAxisDropDownControls() {
	$("#cbX").on('change', function () {
		if ($(this).val() != "Custom Axis") {
			$("#cbX option[value='Custom Axis']").remove();
		}
	});
}

function addHelp() {
	var tooltipText = '<div class="qtip-dark"><b>Task:</b> Your task is to classify all of the points in the scatterplot. Each <i>circle</i> in the scatterplot represents a <i>basketball player</i>. Color each circle according to the <i>position</i> you think the basketball player plays.';
	tooltipText += '<br><br>';
	tooltipText += '<b>Interactions:</b> <ul>';
	tooltipText += '<li><i>See Details</i> about a point by <i>Hovering</i> over it. Details will be shown in the text on the right.</li>';
	tooltipText += '<li><i>Activate a Position</i> by <i>Clicking</i> on a colored circle on the right.</li>';
	tooltipText += '<li><i>Deactivate a Position</i> by <i>Double Clicking</i> on any colored circle on the right.</li>';
	tooltipText += '<li><i>Classify a Point</i> on the scatterplot by <i>Clicking</i> on it while its position is activated.</li>';
	tooltipText += '<li><i>Un-Assign a Point</i> on the scatterplot by <i>Clicking</i> on it while "Un-Assign" is activated.</li>';
	tooltipText += '<li><i>Change the Axes</i> by <i>Selecting</i> a new variable from the drop-down on the X or Y axes.</li>';
	tooltipText += '<li><i>Define a Custom Axis</i> by <i>Dragging</i> points from the scatterplot to the bins along the X-Axis.</li>';
	tooltipText += '<li><i>Remove a Point from a Bin</i> on the X-Axis by <i>Double Clicking</i> the point inside the bin.</li>';
	tooltipText += '<li><i>Reset the X-Axis</i> to the default by <i>Clicking</i> the "Clear X" button to clear both bins and change it to the default dimension.</li>';
	tooltipText += '<li><i>Change the Weight of an Attribute</i> along the X-Axis by <i>Dragging</i> the bars to manually change the weight.</li>';
	tooltipText += '</ul>';
	tooltipText += '<br>';
	tooltipText += 'Try to classify all points in the scatterplot to complete the study. When ready to continue, check the box in bottom right and press <strong><i>Continue</i></strong> to proceed to the next phase of the study.';
	tooltipText += '</div>';
	$("#helpButton").qtip({
		content: {
			title: 'Help:',
			text: tooltipText
		},
		style: {
			width: 500,
			classes: 'qtip-dark'
		}
	});
	d3.select("#helpButton").on("mouseover", function () {
		helpMouseoverStart = new Date();
	}).on("mouseout", function () {
		helpMouseoverEnd = new Date();
		// units are seconds -- mouseover time will always encapsulate drag time as well
		var elapsedTime = (helpMouseoverEnd - helpMouseoverStart) / 1000;

		// get the x,y coordinates of all the points on the graph to log
		var data_locations = [];
		d3.select("#SC").selectAll("circle").each(function (d) {
			var pt_log = { player: d.Name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
			data_locations.push(pt_log);
		});

		ial.logging.log('help', undefined, 'HelpHover', { 'level': 'INFO', 'taskType': 'bball', 'eventType': 'help_hover', 'elapsedTime': elapsedTime, 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });
		R7Insight.log(JSON.stringify(ial.logging.peek()));
		// testing
		console.log(JSON.stringify(ial.logging.peek()));
	});
}

function addClassificationControls(data) {
	var svgClassContainer = d3.select("#datapanel2")
		.append("svg")
		.attr("width", 150)
		.attr("height", 160);
	var categoryGroups = svgClassContainer.selectAll("circle")
		.data(positions)
		.enter().append("g")
		.attr("transform", function (d, i) { return "translate(30, " + (24 * (i + 1)) + ")"; })
		.on("click", function (d) {
			activePosition = d;
			d3.selectAll(".category").classed("categoryClicked", false);
			d3.select(this).select("circle").classed("categoryClicked", true);
			/* ---------- disabling description display ---------- */
			// $("#datapanel3").html("<h5 style=\"color:" + colorMap[d] + "; stroke: 1px black;\">" + d + ":</h5> <h6>" + positionDescriptions[d] + "</h6>");

			// get the x,y coordinates of all the points on the graph to log
			var data_locations = [];
			d3.select("#SC").selectAll("circle").each(function (d) {
				var pt_log = { player: d.Name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
				data_locations.push(pt_log);
			});

			ial.logging.log('category_' + activePosition, undefined, 'CategoryClick', { 'level': 'INFO', 'taskType': 'bball', 'eventType': 'category_click', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });
			// LE.log(JSON.stringify(ial.logging.peek()));
			R7Insight.log(JSON.stringify(ial.logging.peek()));
		}).on("dblclick", function () {
			activePosition = "none";
			d3.selectAll(".category").classed("categoryClicked", false);
			$("#datapanel3").html("");

			// get the x,y coordinates of all the points on the graph to log
			var data_locations = [];
			d3.select("#SC").selectAll("circle").each(function (d) {
				var pt_log = { player: d.Name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
				data_locations.push(pt_log);
			});

			ial.logging.log('category_' + activePosition, undefined, 'CategoryDoubleClick', { 'level': 'INFO', 'taskType': 'bball', 'eventType': 'category_double_click', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });
			// LE.log(JSON.stringify(ial.logging.peek()));
			R7Insight.log(JSON.stringify(ial.logging.peek()));
		});
	var categoryCircles = categoryGroups.append("circle")
		.attr("cx", 0)
		.attr("cy", function (d, i) { return 0; })
		.attr("r", 10)
		.classed("category", true)
		.style("fill", function (d) { return colorMap[d]; });
	var categoryText = categoryGroups.append("text")
		.attr("dx", function (d) { return 20; })
		.text(function (d) { return d; })
		.attr("dominant-baseline", "middle");

	// listen for user to hit Continue
	$("#continueButton").click(function () {
		if (document.getElementById('doneCheck').checked) {
			var allClassified = true;
			var howMany = 0;
			for (var i = 0; i < data.length; i++) {
				if (data[i]["coord"]["userlabel"] == label.DEFAULT) {
					allClassified = false;
					howMany++;
				}
			}
			var prevURL = document.referrer;
			if (localStorage.getItem("first_task") == 'bball') {
				if (!allClassified) {
					if ((data.length - howMany) <= (data.length * 0.8)) {
						var userResp = alert("Please classify more points.");
					} else {
						var userResp = confirm(howMany + " out of " + data.length + " points have not been classified yet. Are you sure you want to continue?");
						if (userResp) { // user wants to continue anyway
							var w = window.open("custom_axis_dog.html", "_self");
						}
					}
				} else {
					var w = window.open("custom_axis_dog.html", "_self");
					window.localStorage.setItem("userId", window.localStorage.getItem("userId"));
					window.localStorage.setItem("whichCondition", window.localStorage.getItem("whichCondition"));
				}
			} else {
				if (!allClassified) {
					if ((data.length - howMany) <= (data.length * 0.8)) {
						var userResp = alert("Please classify more points.");
					} else {
						var userResp = confirm(howMany + " out of " + data.length + " points have not been classified yet. Are you sure you want to continue?");
						if (userResp) { // user wants to continue anyway
							var w = window.open("postsurvey.html", "_self");
						}
					}
				} else {
					var w = window.open("postsurvey.html", "_self");
					window.localStorage.setItem("userId", window.localStorage.getItem("userId"));
					window.localStorage.setItem("whichCondition", window.localStorage.getItem("whichCondition"));
				}
			}
		} else alert("Check the box to certify you are finished with the task.");
	});
}

function drawScatterPlot(data) {

	// heterogeneous data
	initdim1 = attr.indexOf(localStorage.getItem("defaultX_bball")), initdim2 = attr.indexOf(localStorage.getItem("defaultY_bball")); // 1, 2 = height, weight
	data.forEach(function (d) { d.x = d["coord"][attr[initdim1]]; d.y = d["coord"][attr[initdim2]]; });
	graph = new SimpleGraph("scplot", data, {
		"xlabel": attr[initdim1],
		"ylabel": attr[initdim2],
		"init": true
	});

	var V1 = {}, V2 = {};
	for (var i = 0; i < attrNo; i++) {
		X[i] = { "attr": attr[i], "value": 0, "changed": 0, "error": 0 };
		Y[i] = { "attr": attr[i], "value": 0, "changed": 0, "error": 0 };
		V1[attr[i]] = 0;
		V2[attr[i]] = 0;
	}

	V1[attr[initdim1]] = 1;
	V2[attr[initdim2]] = 1;

	// get the x,y coordinates of all the points on the graph to log
	var data_locations = [];
	d3.select("#SC").selectAll("circle").each(function (d) {
		var pt_log = { player: d.Name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
		data_locations.push(pt_log);
	});

	//update IAL weight vector
	ial.usermodel.setAttributeWeightVector(V1, true, { 'level': 'INFO', 'taskType': 'bball', 'eventType': 'set_attribute_weight_vector_init', 'whichAxis': 'X', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });
	// LE.log(JSON.stringify(ial.logging.peek()));
	R7Insight.log(JSON.stringify(ial.logging.peek()));
	ial.usermodel.setAttributeWeightVector(V2, true, { 'level': 'INFO', 'taskType': 'bball', 'eventType': 'set_attribute_weight_vector_init', 'whichAxis': 'Y', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });
	// LE.log(JSON.stringify(ial.logging.peek()));
	R7Insight.log(JSON.stringify(ial.logging.peek()));

	X[initdim1]["value"] = 1;
	Y[initdim2]["value"] = 1;
	document.getElementById("cbX").selectedIndex = initdim1;
	document.getElementById("cbY").selectedIndex = initdim2;

	xaxis = new axis("#scplot", X, "X", {
		"width": graph.size.width - dropSize * 2,
		"height": graph.padding.bottom - 40,
		"padding": { top: graph.padding.top + graph.size.height + 40, right: 0, left: graph.padding.left + dropSize + 10, bottom: 0 }
	});
}