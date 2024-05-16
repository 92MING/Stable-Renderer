import { api } from "./api.js"
import "./domWidget.js";

let controlValueRunBefore = false;
export function updateControlWidgetLabel(widget) {
	let replacement = "after";
	let find = "before";
	if (controlValueRunBefore) {
		[find, replacement] = [replacement, find]
	}
	widget.label = (widget.label ?? widget.name).replace(find, replacement);
}

const IS_CONTROL_WIDGET = Symbol();
const HAS_EXECUTED = Symbol();

function getNumberDefaults(inputData, defaultStep, precision, enable_rounding) {
	let number_type = inputData[0];
	let is_float = number_type.toLowerCase() == "float";

	let defaultVal = inputData?.[1]?.["default"]?? 0;
	let min = inputData?.[1]?.["min"]?? 0;
	let max = inputData?.[1]?.["max"]?? 2048;
	let step = inputData?.[1]?.["step"]?? defaultStep?? null;
	if (step == null){
		if (is_float) {
			step = 0.1;  // float number has default step 0.1
		} else {
			step = 1;	 // integer number has default step 1
		}
	}
	let round = inputData?.[1]?.["round"]?? null;
	if (round == null) {
		if (is_float) {
			round = 0.01;  // float number has default round 0.01
		} else {
			round = 1;	 // integer number has default round 0
		}
	}

	// precision is the number of decimal places to show.
	// by default, display the the smallest number of decimal places such that changes of size step are visible.
	if (precision == undefined || precision == null) {
		precision = Math.max(-Math.floor(Math.log10(step)),0);
	}

	if (enable_rounding && (round == undefined || round==null || round === true)) {
		// by default, round the value to those decimal places shown.
		round = Math.round(1000000*Math.pow(0.1,precision))/1000000;
	}

	return { val: defaultVal, config: { min, max, step: 10.0 * step, round, precision } };
}

export function addValueControlWidget(node, targetWidget, defaultValue = "randomize", values, widgetName, inputData) {
	let name = inputData[1]?.control_after_generate;
	if(typeof name !== "string") {
		name = widgetName;
	}
	const widgets = addValueControlWidgets(node, targetWidget, defaultValue, {
		addFilterList: false,
		controlAfterGenerateName: name
	}, inputData);
	return widgets[0];
}

export function addValueControlWidgets(node, targetWidget, defaultValue = "randomize", options, inputData) {
	if (!defaultValue) defaultValue = "randomize";
	if (!options) options = {};

	const getName = (defaultName, optionName) => {
		let name = defaultName;
		if (options[optionName]) {
			name = options[optionName];
		} else if (typeof inputData?.[1]?.[defaultName] === "string") {
			name = inputData?.[1]?.[defaultName];
		} else if (inputData?.[1]?.control_prefix) {
			name = inputData?.[1]?.control_prefix + " " + name
		}
		return name;
	}

	const widgets = [];
	const valueControl = node.addWidget(
		"combo",
		getName("control_after_generate", "controlAfterGenerateName"),
		defaultValue,
		function () {},
		{
			values: ["fixed", "increment", "decrement", "randomize"],
			serialize: false, // Don't include this in prompt.
		}
	);
	valueControl[IS_CONTROL_WIDGET] = true;
	updateControlWidgetLabel(valueControl);
	widgets.push(valueControl);

	const isCombo = targetWidget.type === "combo";
	let comboFilter;
	if (isCombo) {
		valueControl.options.values.push("increment-wrap");
	}
	if (isCombo && options.addFilterList !== false) {
		comboFilter = node.addWidget(
			"string",
			getName("control_filter_list", "controlFilterListName"),
			"",
			function () {},
			{
				serialize: false, // Don't include this in prompt.
			}
		);
		updateControlWidgetLabel(comboFilter);

		widgets.push(comboFilter);
	}

	const applyWidgetControl = () => {
		var v = valueControl.value;

		if (isCombo && v !== "fixed") {
			let values = targetWidget.options.values;
			const filter = comboFilter?.value;
			if (filter) {
				let check;
				if (filter.startsWith("/") && filter.endsWith("/")) {
					try {
						const regex = new RegExp(filter.substring(1, filter.length - 1));
						check = (item) => regex.test(item);
					} catch (error) {
						console.error("Error constructing RegExp filter for node " + node.id, filter, error);
					}
				}
				if (!check) {
					const lower = filter.toLocaleLowerCase();
					check = (item) => item.toLocaleLowerCase().includes(lower);
				}
				values = values.filter(item => check(item));
				if (!values.length && targetWidget.options.values.length) {
					console.warn("Filter for node " + node.id + " has filtered out all items", filter);
				}
			}
			let current_index = values.indexOf(targetWidget.value);
			let current_length = values.length;

			switch (v) {
				case "increment":
					current_index += 1;
					break;
				case "increment-wrap":
					current_index += 1;
					if ( current_index >= current_length ) {
					    current_index = 0;
					}
					break;
				case "decrement":
					current_index -= 1;
					break;
				case "randomize":
					current_index = Math.floor(Math.random() * current_length);
				default:
					break;
			}
			current_index = Math.max(0, current_index);
			current_index = Math.min(current_length - 1, current_index);
			if (current_index >= 0) {
				let value = values[current_index];
				targetWidget.value = value;
				targetWidget.callback(value);
			}
		} else {
			//number
			let min = targetWidget.options.min;
			let max = targetWidget.options.max;
			// limit to something that javascript can handle
			max = Math.min(1125899906842624, max);
			min = Math.max(-1125899906842624, min);
			let range = (max - min) / (targetWidget.options.step / 10);

			//adjust values based on valueControl Behaviour
			switch (v) {
				case "fixed":
					break;
				case "increment":
					targetWidget.value += targetWidget.options.step / 10;
					break;
				case "decrement":
					targetWidget.value -= targetWidget.options.step / 10;
					break;
				case "randomize":
					targetWidget.value = Math.floor(Math.random() * range) * (targetWidget.options.step / 10) + min;
				default:
					break;
			}
			/*check if values are over or under their respective
			 * ranges and set them to min or max.*/
			if (targetWidget.value < min) targetWidget.value = min;

			if (targetWidget.value > max)
				targetWidget.value = max;
			targetWidget.callback(targetWidget.value);
		}
	};

	valueControl.beforeQueued = () => {
		if (controlValueRunBefore) {
			// Don't run on first execution
			if (valueControl[HAS_EXECUTED]) {
				applyWidgetControl();
			}
		}
		valueControl[HAS_EXECUTED] = true;
	};

	valueControl.afterQueued = () => {
		if (!controlValueRunBefore) {
			applyWidgetControl();
		}
	};

	return widgets;
};

function seedWidget(node, inputName, inputData, app, widgetName) {
	const seed = createIntWidget(node, inputName, inputData, app, true);
	const seedControl = addValueControlWidget(node, seed.widget, "randomize", undefined, widgetName, inputData);

	seed.widget.linkedWidgets = [seedControl];
	return seed;
}

function createIntWidget(node, inputName, inputData, app, isSeedInput) {
	const control = inputData[1]?.control_after_generate;
	if (!isSeedInput && control) {
		return seedWidget(node, inputName, inputData, app, typeof control === "string" ? control : undefined);
	}

	var extra_params = inputData[1]? inputData[1] : {};
	var display_mode = extra_params?.display?? "number";

	let widgetType = isSlider(display_mode, app);
	const { val, config } = getNumberDefaults(inputData, 1, 0, true);
	Object.assign(config, { precision: 0 });
	return {
		widget: node.addWidget(
			widgetType,
			inputName,
			val,
			function (v) {
				const s = this.options.step / 10;
				this.value = Math.round(v / s) * s;
			},
			config
		),
	};
}

/**
 * Uploads a file to the server and returns the path to the file.
 * @param {File} file - The file to upload.
 * @param {string} save_to - The folder to save the file to: ['input', 'output', 'temp']
*/
async function uploadFile(file, save_to='input') {
	try {
	  // Wrap file in formdata so it includes filename
	  const body = new FormData();
	  const new_file = new File([file], file.name, {
		type: file.type,
		lastModified: file.lastModified,
	  });
	  body.append("data", new_file);
	  body.append("type", save_to);
	  const resp = await api.fetchApi("/upload/file", {
		method: "POST",
		body,
	  });
  
	  if (resp.status === 200 || resp.status === 201) {
		return resp.json();
	  } else {
		alert(`Upload failed: ${resp.statusText}`);
	  }
	} catch (error) {
	  alert(`Upload failed: ${error}`);
	}
  }

function createPathWidget(node, inputName, inputData, app, widgetTypeName="path") {
	var widget = node.addWidget(
		widgetTypeName, 
		inputName, 
		"", 
		function (canvas, node){
			// find if document has a file input element, if not, create one
			var fileInput = document.getElementById("comfy_file_input");
			if (!fileInput) {
				var fileInput = document.createElement("input");
				fileInput.id = "comfy_file_input";
				fileInput.type = "file";
				fileInput.style.display = "none";
				document.body.appendChild(fileInput);
			}

			var accept_types = inputData[1]?.accept_types;
			accept_types = accept_types ?? "*";
			fileInput.accept = accept_types;	// set the accept types for the file input
			
			var accept_multiple = inputData[1]?.accept_multiple;
			accept_multiple = accept_multiple ?? false;

			var accept_folder = inputData[1]?.accept_folder;
			accept_folder = accept_folder ?? false;
			
			var to_folder = inputData[1]?.to_folder;
			to_folder = to_folder ?? "input";
			
			if (accept_folder) {
				fileInput.setAttribute("webkitdirectory", "");
				fileInput.setAttribute("directory", "");
			}
			else {
				if (fileInput.hasAttribute("webkitdirectory"))
					fileInput.removeAttribute("webkitdirectory");
				if (fileInput.hasAttribute("directory"))
					fileInput.removeAttribute("directory");
			}
			if (accept_multiple){
				fileInput.setAttribute("multiple", "");
			}
			
			// add an event listener to the file input element
			fileInput.onchange = async function() {
				if (accept_multiple){
					let vals = [];
					var async_upload = async function(file, i) {
						var resp = await uploadFile(file, to_folder);
						vals[i] = resp.absolute_path;
					}
					await Promise.all(Array.from(fileInput.files).map(async_upload));
					this.value = vals.join(';');
				}
				else {
					const file = fileInput.files[0];
					if (file) {
						var resp = await uploadFile(file, to_folder);
						var abs_path = resp.absolute_path;
						this.value = abs_path;
					}
				}
			}.bind(this);

			// trigger a click event on the file input element
			fileInput.click();
		},
		{}
	);
	return { widget };
}
function createPathsWidget(node, inputName, inputData, app){
	return createPathWidget(node, inputName, inputData, app, "paths");
}

function addMultilineWidget(node, name, opts, app) {
	const inputEl = document.createElement("textarea");
	inputEl.className = "comfy-multiline-input";
	inputEl.value = opts.defaultVal;
	inputEl.placeholder = opts.placeholder || name;

	const widget = node.addDOMWidget(name, "customtext", inputEl, {
		getValue() {
			return inputEl.value;
		},
		setValue(v) {
			inputEl.value = v;
		},
	});
	widget.inputEl = inputEl;

	inputEl.addEventListener("input", () => {
		widget.callback?.(widget.value);
	});

	return { minWidth: 400, minHeight: 200, widget };
}

function isSlider(display, app) {
	if (app.ui.settings.getSettingValue("Comfy.DisableSliders")) {
		return "number"
	}

	return (display==="slider") ? "slider" : "number"
}

export function initWidgets(app) {
	app.ui.settings.addSetting({
		id: "Comfy.WidgetControlMode",
		name: "Widget Value Control Mode",
		type: "combo",
		defaultValue: "after",
		options: ["before", "after"],
		tooltip: "Controls when widget values are updated (randomize/increment/decrement), either before the prompt is queued or after.",
		onChange(value) {
			controlValueRunBefore = value === "before";
			for (const n of app.graph._nodes) {
				if (!n.widgets) continue;
				for (const w of n.widgets) {
					if (w[IS_CONTROL_WIDGET]) {
						updateControlWidgetLabel(w);
						if (w.linkedWidgets) {
							for (const l of w.linkedWidgets) {
								updateControlWidgetLabel(l);
							}
						}
					}
				}
			}
			app.graph.setDirtyCanvas(true);
		},
	});
}

export const ComfyWidgets = {
	"INT:seed": seedWidget,
	"INT:noise_seed": seedWidget,
	FLOAT(node, inputName, inputData, app) {
		let input_extra_params = inputData[1]? inputData[1] : {};
		let display_mode = input_extra_params?.display?? "number";

		let widgetType = isSlider(display_mode, app);

		let precision = app.ui.settings.getSettingValue("Comfy.FloatRoundingPrecision");
		let disable_rounding = app.ui.settings.getSettingValue("Comfy.DisableFloatRounding")
		
		if (precision == 0) precision = undefined;
		const { val, config } = getNumberDefaults(inputData, 0.5, precision, !disable_rounding);
		return { widget: node.addWidget(widgetType, inputName, val,
			function (v) {
				if (config.round) {
					this.value = Math.round(v/config.round)*config.round;
				} else {
					this.value = v;
				}
			}, config) };
	},
	INT(node, inputName, inputData, app) {
		return createIntWidget(node, inputName, inputData, app);
	},
	BOOLEAN(node, inputName, inputData) {
		let options = {};

		let extra_params = inputData[1]? inputData[1] : {};
		let default_value = extra_params?.default?? false;
		let label_on = extra_params?.label_on?? "True";
		let label_off = extra_params?.label_off?? "False";

		options["on"] = label_on;
		options["off"] = label_off;

		return {
			widget: node.addWidget(
				"toggle",
				inputName,
				default_value,
				() => {},
				options,
				)
		};
	},
	STRING(node, inputName, inputData, app) {
		let input_extra_params = inputData[1]? inputData[1] : {};
		var defaultVal = input_extra_params["default"]?? "";
		var multiline = input_extra_params["multiline"]? !!input_extra_params["multiline"] : false;
		
		let res;
		if (multiline) {
			res = addMultilineWidget(node, inputName, { defaultVal, ...inputData[1] }, app);
		} else {
			res = { widget: node.addWidget("text", inputName, defaultVal, () => {}, {}) };
		}

		if(inputData[1].dynamicPrompts != undefined && inputData[1].dynamicPrompts != null)
			res.widget.dynamicPrompts = inputData[1].dynamicPrompts;

		return res;
	},
	COMBO(node, inputName, inputData) {
		const type = inputData[0];
		let defaultValue = type[0];
		if (inputData[1] && inputData[1].default) {
			defaultValue = inputData[1].default;
		}
		const res = { widget: node.addWidget("combo", inputName, defaultValue, () => {}, { values: type }) };
		if (inputData[1]?.control_after_generate) {
			res.widget.linkedWidgets = addValueControlWidgets(node, res.widget, undefined, undefined, inputData);
		}
		return res;
	},
	IMAGEUPLOAD(node, inputName, inputData, app) {
		const imageWidget = node.widgets.find((w) => w.name === (inputData[1]?.widget ?? "image"));
		let uploadWidget;

		function showImage(name) {
			const img = new Image();
			img.onload = () => {
				node.imgs = [img];
				app.graph.setDirtyCanvas(true);
			};
			let folder_separator = name.lastIndexOf("/");
			let subfolder = "";
			if (folder_separator > -1) {
				subfolder = name.substring(0, folder_separator);
				name = name.substring(folder_separator + 1);
			}
			img.src = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=input&subfolder=${subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`);
			node.setSizeForImage?.();
		}

		var default_value = imageWidget.value;
		Object.defineProperty(imageWidget, "value", {
			set : function(value) {
				this._real_value = value;
			},

			get : function() {
				let value = "";
				if (this._real_value) {
					value = this._real_value;
				} else {
					return default_value;
				}

				if (value.filename) {
					let real_value = value;
					value = "";
					if (real_value.subfolder) {
						value = real_value.subfolder + "/";
					}

					value += real_value.filename;

					if(real_value.type && real_value.type !== "input")
						value += ` [${real_value.type}]`;
				}
				return value;
			}
		});

		// Add our own callback to the combo widget to render an image when it changes
		const cb = node.callback;
		imageWidget.callback = function () {
			showImage(imageWidget.value);
			if (cb) {
				return cb.apply(this, arguments);
			}
		};

		// On load if we have a value then render the image
		// The value isnt set immediately so we need to wait a moment
		// No change callbacks seem to be fired on initial setting of the value
		requestAnimationFrame(() => {
			if (imageWidget.value) {
				showImage(imageWidget.value);
			}
		});

		async function uploadFile(file, updateNode, pasted = false) {
			try {
				// Wrap file in formdata so it includes filename
				const body = new FormData();
				body.append("image", file);
				if (pasted) body.append("subfolder", "pasted");
				const resp = await api.fetchApi("/upload/image", {
					method: "POST",
					body,
				});

				if (resp.status === 200) {
					const data = await resp.json();
					// Add the file to the dropdown list and update the widget value
					let path = data.name;
					if (data.subfolder) path = data.subfolder + "/" + path;

					if (!imageWidget.options.values.includes(path)) {
						imageWidget.options.values.push(path);
					}

					if (updateNode) {
						showImage(path);
						imageWidget.value = path;
					}
				} else {
					alert(resp.status + " - " + resp.statusText);
				}
			} catch (error) {
				alert(error);
			}
		}

		const fileInput = document.createElement("input");
		Object.assign(fileInput, {
			type: "file",
			accept: "image/jpeg,image/png,image/webp",
			style: "display: none",
			onchange: async () => {
				if (fileInput.files.length) {
					await uploadFile(fileInput.files[0], true);
				}
			},
		});
		document.body.append(fileInput);

		// Create the button widget for selecting the files
		uploadWidget = node.addWidget("button", inputName, "image", () => {
			fileInput.click();
		});
		uploadWidget.label = "choose file to upload";
		uploadWidget.serialize = false;

		// Add handler to check if an image is being dragged over our node
		node.onDragOver = function (e) {
			if (e.dataTransfer && e.dataTransfer.items) {
				const image = [...e.dataTransfer.items].find((f) => f.kind === "file");
				return !!image;
			}

			return false;
		};

		// On drop upload files
		node.onDragDrop = function (e) {
			console.log("onDragDrop called");
			let handled = false;
			for (const file of e.dataTransfer.files) {
				if (file.type.startsWith("image/")) {
					uploadFile(file, !handled); // Dont await these, any order is fine, only update on first one
					handled = true;
				}
			}

			return handled;
		};

		node.pasteFile = function(file) {
			if (file.type.startsWith("image/")) {
				const is_pasted = (file.name === "image.png") &&
								  (file.lastModified - Date.now() < 2000);
				uploadFile(file, true, is_pasted);
				return true;
			}
			return false;
		}

		return { widget: uploadWidget };
	},
	PATH(node, inputName, inputData, app) {
		return createPathWidget(node, inputName, inputData, app);
	},
	PATHS(node, inputName, inputData, app) {
		return createPathsWidget(node, inputName, inputData, app);
	}
};
