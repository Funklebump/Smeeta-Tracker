# smeeta-tracker-2
Track smeeta affinity procs with an overlay and sound notifications. Contains additional tools for tracking Vitus Essence in arbitration runs.

This application is fan made and not endorsed by Digital Extremes. This does nothing that is against TOS, but use your own judgement and use at your own risk.

# Smeeta detector tips
Although generally unnecessary, to improve accuracy you can change your UI colors for "Text" and "Buffs" to uncommon and highly saturated colors. The default text color is white, which is completely unsaturated and moreover fairly common in many tilesets. This is not ideal, but the app tries its best to compensate for this. This is the only supported unsaturated color since it is the default. The default buff icon color is a highly saturated blue, and the main issue is that blue is a common color for the sky. On some maps this will cause template matching to fail. You can disable template matching, but I do not recommend it. To do this, set the template matching scale in the settings tab to 0. A warning though, if you disable this it will only be able to detect procs based on their duration. Some warframe abilities can reach 120 seconds or more, causing false detection. The number of false detections may increase as well because the template matching acts as a filter. If you have buff abilities with very high duration it is best to leave template matching on. 

Unfortunately the smeeta buff icon is slightly transparent, making the color you have to filter for different based on the background. This poses some difficulty for a robust detection mechanism, and hopefully there is something that can be done in the future to fix the cases where the current algorithm fails. I have noticed that problems can arise on the new Corpus tilesets and Mars tilesets (because of the blue sky).

There are too many colors to test, so I cannot gurantee all of them working, but the best color I can recommend is in the Cherub palette: row 14, column 5 (index starting from 1). If you don't have this palette, another option is in classic saturated: row 4 column 5. Of course this comes with the downside that your UI will be a saturated pink and be hardly human readable, so it is up to you. You can also experiment with other color options of course that do not make your eyes bleed.

Currently this only supports a UI scale of 100%. Support for higher values will come in the future, and having a higher value will likely further improve detection accuracy. (It is already very accurate though)

Lastly, your Video settings can have an effect on detections, namely "Brightness" and "Contrast". I would leave each of these at 50%. It is possible that other settings work, but I just have not tested them. I believe if you are using the default text color (white), lower brightness and higher contrast will be better for detections.


# Arbitration Logger
THIS DOES NOT WORK FOR CLIENTS. YOU HAVE TO BE HOSTING FOR THIS INFORMATION TO BE AVAILABLE

This is just how the game works, nothing I can do to fix this.

